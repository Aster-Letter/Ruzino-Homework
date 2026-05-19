#include "iisph.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace USTC_CG::sph_fluid {

using namespace Eigen;

namespace {
constexpr double kDensityEpsilon = 1e-12;
constexpr double kAiiEpsilon = 1e-9;
constexpr int kParallelParticleThreshold = 256;
}

IISPH::IISPH(const MatrixXd& X, const Vector3d& box_min, const Vector3d& box_max)
    : SPHBase(X, box_min, box_max)
{
    resize_solver_storage();
}

// Centrally call necessary functions to compute one step of IISPH.
void IISPH::step()
{
    TIC(step)

    ps_.assign_particles_to_cells();
    ps_.search_neighbors();

    compute_density();
    compute_non_pressure_acceleration();

    predict_advection();
    compute_pressure();

    compute_pressure_gradient_acceleration();
    advect();
    update_diagnostics(last_pressure_iterations_, last_pressure_residual_);

    TOC(step)
}

void IISPH::compute_pressure()
{
    auto& particles = ps_.particles();
    const int n = static_cast<int>(particles.size());
    resize_solver_storage();

#pragma omp parallel for if(n > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n; ++i) {
        particles[i]->pressure() = std::max(0.0, last_pressure_[i]);
    }

    const double threshold = 0.001;
    last_pressure_iterations_ = 0;
    last_pressure_residual_ = 0.0;

    for (int iter = 0; iter < max_iter_; ++iter) {
        last_pressure_residual_ = pressure_solve_iteration();
        last_pressure_iterations_ = iter + 1;

        if (last_pressure_residual_ < threshold) {
            break;
        }
    }

#pragma omp parallel for if(n > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n; ++i) {
        last_pressure_[i] = particles[i]->pressure();
    }
}

void IISPH::predict_advection()
{
    auto& particles = ps_.particles();
    const double m = ps_.mass();
    const double h = ps_.h();
    const int n = static_cast<int>(particles.size());
    resize_solver_storage();

    predict_density_.setZero(n);
    aii_.setZero(n);

#pragma omp parallel for if(n > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n; ++i) {
        const auto& p = particles[i];

        const double rho_i = p->density();
        if (rho_i <= kDensityEpsilon) {
            predict_density_[i] = ps_.density0();
            aii_[i] = 0.0;
            continue;
        }

        const Vector3d v_i_star = p->vel() + dt_ * p->acceleration();

        double div_v = 0.0;

        for (const auto& q : p->neighbors()) {
            const double rho_j = q->density();
            if (rho_j <= kDensityEpsilon) {
                continue;
            }

            const Vector3d v_j_star = q->vel() + dt_ * q->acceleration();
            const Vector3d grad = grad_W(p->x() - q->x(), h);

            div_v += (m / rho_j) * (v_j_star - v_i_star).dot(grad);
        }

        predict_density_[i] = rho_i - dt_ * rho_i * div_v;

        Vector3d d_ii = Vector3d::Zero();
        for (const auto& q : p->neighbors()) {
            const Vector3d grad = grad_W(p->x() - q->x(), h);
            d_ii += m / (rho_i * rho_i) * grad;
        }

        double aii = 0.0;
        for (const auto& q : p->neighbors()) {
            const Vector3d grad_ij = grad_W(p->x() - q->x(), h);
            const Vector3d d_ji = m / (rho_i * rho_i) *
                                  grad_W(q->x() - p->x(), h);

            aii -= m * (d_ii - d_ji).dot(grad_ij);
        }

        aii_[i] = aii;
    } 
}

double IISPH::pressure_solve_iteration()
{
    auto& particles = ps_.particles();
    const int n = static_cast<int>(particles.size());
    const double h = ps_.h();
    const double m = ps_.mass();
    const double rho0 = ps_.density0();
    resize_solver_storage();

#pragma omp parallel for if(n > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n; ++i) {
        pressure_old_[i] = particles[i]->pressure();
        pressure_acc_[i] = Vector3d::Zero();
    }

#pragma omp parallel for if(n > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n; ++i) {
        const auto& p = particles[i];

        const double rho_i = p->density();
        const double p_i = pressure_old_[i];
        if (rho_i <= kDensityEpsilon) {
            continue;
        }

        Vector3d acc = Vector3d::Zero();

        for (const auto& q : p->neighbors()) {
            const int j = q->idx();

            const double rho_j = q->density();
            if (rho_j <= kDensityEpsilon) {
                continue;
            }
            const double p_j = pressure_old_[j];

            const Vector3d grad = grad_W(p->x() - q->x(), h);

            const double coeff =
                m * (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j));

            acc -= coeff * grad;
        }

        pressure_acc_[i] = acc;
    }

    double density_error_sum = 0.0;
#pragma omp parallel for if(n > kParallelParticleThreshold) schedule(static) reduction(+:density_error_sum)
    for (int i = 0; i < n; ++i) {
        const auto& p = particles[i];

        double Api = 0.0;
        for (const auto& q : p->neighbors()) {
            const int j = q->idx();

            const Vector3d grad = grad_W(p->x() - q->x(), h);

            Api += m * (pressure_acc_[i] - pressure_acc_[j]).dot(grad);
        }

        Api_[i] = Api;

        const double b_i = (rho0 - predict_density_[i]) / (dt_ * dt_);

        double new_pressure = pressure_old_[i];

        if (std::abs(aii_[i]) > kAiiEpsilon) {
            new_pressure =
                pressure_old_[i] + omega_ * (b_i - Api_[i]) / aii_[i];
        }

        new_pressure = std::clamp(new_pressure, 0.0, pressure_clamp_);

        p->pressure() = new_pressure;

        const double rho_after_pressure =
            predict_density_[i] + dt_ * dt_ * Api_[i];

        density_error_sum += std::abs(rho_after_pressure - rho0); 
    }

    return n > 0 ? density_error_sum / (n * rho0) : 0.0; 
}

// ------------------ helper function, no need to modify ---------------------
void IISPH::reset()
{
    SPHBase::reset();
    resize_solver_storage();
    predict_density_.setZero();
    aii_.setZero();
    Api_.setZero();
    last_pressure_.setZero();
    pressure_old_.setZero();
    std::fill(pressure_acc_.begin(), pressure_acc_.end(), Vector3d::Zero());
    last_pressure_iterations_ = 0;
    last_pressure_residual_ = 0.0;
}

void IISPH::resize_solver_storage()
{
    const int n = static_cast<int>(ps_.particles().size());
    if (predict_density_.size() == n && aii_.size() == n &&
        Api_.size() == n && last_pressure_.size() == n &&
        pressure_old_.size() == n &&
        static_cast<int>(pressure_acc_.size()) == n) {
        return;
    }

    predict_density_ = VectorXd::Zero(n);
    aii_ = VectorXd::Zero(n);
    Api_ = VectorXd::Zero(n);
    last_pressure_ = VectorXd::Zero(n);
    pressure_old_ = VectorXd::Zero(n);
    pressure_acc_.assign(n, Vector3d::Zero());
}
}  // namespace USTC_CG::node_sph_fluid
