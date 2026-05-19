#include "sph_base.h"
#include <algorithm>
#include <cmath>
#define M_PI 3.14159265358979323846
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <limits>
#include "colormap_jet.h"

namespace USTC_CG::sph_fluid {
using namespace Eigen;
using Real = double;

namespace {
constexpr int kParallelParticleThreshold = 256;
}

SPHBase::SPHBase(const Eigen::MatrixXd& X, const Vector3d& box_min, const Vector3d& box_max)
    : init_X_(X),
      X_(X),
      vel_(MatrixXd::Zero(X.rows(), X.cols())),
      box_max_(box_max),
      box_min_(box_min),
      ps_(X, box_min, box_max)
{
}

// ----------------- SPH kernal function and its spatial derivatives, no need to modify -----------------
double SPHBase::W(const Eigen::Vector3d& r, double h)
{
    double h3 = h * h * h;
    double m_k = 8.0 / (M_PI * h3);
    double m_l = 48.0 / (M_PI * h3); 
    const double q = r.norm() / h;
    double result = 0.;

    if (q <= 1.0) {
        if (q <= 0.5) {
            const Real q2 = q * q;
            const Real q3 = q2 * q;
            result = m_k * (6.0 * q3 - 6.0 * q2 + 1.0);
        }
        else {
            result = m_k * (2.0 * pow(1.0 - q, 3.0));
        }
    }
    return result;
}

double SPHBase::W_zero(double h)
{
    double h3 = h * h * h;
    double m_k = 8.0 / (M_PI * h3);
    return m_k;
}

Vector3d SPHBase::grad_W(const Vector3d& r, double h)
{
    double h3 = h * h * h;
    double m_k = 8.0 / (M_PI * h3);
    double m_l = 48.0 / (M_PI * h3);

    const double rl = r.norm();
    const double q = rl / h;
    Vector3d result = Vector3d::Zero();

    if (q <= 1.0 && rl > 1e-9) {
        Vector3d grad_q = r / rl;
        if (q <= 0.5) {
            result = m_l * q * (3.0 * q - 2.0) * grad_q;
        }
        else {
            const Real factor = 1.0 - q;
            result = -m_l * factor * factor * grad_q;
        }
    }
    return result;
}
// ---------------------------------------------------------------------------------------


void SPHBase::compute_density()
{
    auto& particles = ps_.particles();

    const double mass = ps_.mass();
    const double h = ps_.h();
    const double w0 = W_zero(h);
    const int n_particles = static_cast<int>(particles.size());

#pragma omp parallel for if(n_particles > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n_particles; i++) {
        const auto& p = particles[i];

        double density = mass * w0;
        for (const auto& q : p->neighbors()) {
            density += mass * W(p->x() - q->x(), h);
        }
        p->density() = density;
    }
}

void SPHBase::compute_pressure()
{
    // Not implemented, should be implemented in children classes WCSPH, IISPH, etc. 
}

void SPHBase::compute_non_pressure_acceleration()
{
    auto& particles = ps_.particles();
    const int n_particles = static_cast<int>(particles.size());

#pragma omp parallel for if(n_particles > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n_particles; i++) {
        const auto& p = particles[i];
        Vector3d acceleration = gravity_;

        for (const auto& q : p->neighbors()) {
            acceleration += compute_viscosity_acceleration(p, q);
        }

        p->acceleration() = acceleration;
    }
}

// compute viscosity acceleration between two particles
Vector3d SPHBase::compute_viscosity_acceleration(
    const std::shared_ptr<Particle>& p,
    const std::shared_ptr<Particle>& q)
{
    const Vector3d v_ij = p->vel() - q->vel();
    const Vector3d x_ij = p->x() - q->x();

    const double h = ps_.h();
    const Vector3d grad = grad_W(x_ij, h);
    const double denom = x_ij.squaredNorm() + 0.01 * h * h;
    const double mass = ps_.mass();
    const double density_j = q->density();
    const int dim = 3;

    if (density_j <= 1e-12) {
        return Vector3d::Zero();
    }

    const Vector3d laplace_v =
        2.0 * (dim + 2.0) * (mass / density_j) *
        (v_ij.dot(x_ij) / denom) * grad;
    return viscosity_ * laplace_v;
}

// Traverse all particles and compute pressure gradient acceleration
void SPHBase::compute_pressure_gradient_acceleration()
{
    auto& particles = ps_.particles();
    const int n_particles = static_cast<int>(particles.size());
    const double mass = ps_.mass();
    const double h = ps_.h();

#pragma omp parallel for if(n_particles > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n_particles; i++) {
        const auto& p = particles[i];
        Vector3d acc_p = Vector3d::Zero();

        const double rho_i = p->density();
        const double p_i = p->pressure();
        if (rho_i <= 1e-12) {
            continue;
        }

        for (const auto& q : p->neighbors()) {
            const double rho_j = q->density();
            const double p_j = q->pressure();
            if (rho_j <= 1e-12) {
                continue;
            }

            const Vector3d grad = grad_W(p->x() - q->x(), h);

            const double coeff =
                mass *
                (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j));

            acc_p -= coeff * grad;
        }
        p->acceleration() += acc_p;
    }
}

void SPHBase::step()
{
    // Not implemented, should be implemented in children classes WCSPH, IISPH, etc. 
}


void SPHBase::advect()
{
    auto& particles = ps_.particles();
    const int n_particles = static_cast<int>(particles.size());

#pragma omp parallel for if(n_particles > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n_particles; i++) {
        const auto& p = particles[i];
        p->vel() += p->acceleration() * dt_;
        p->x() += p->vel() * dt_;

        check_collision(p);

        vel_.row(p->idx()) = p->vel().transpose();
        X_.row(p->idx()) = p->x().transpose();
    }
}

// ------------------------------- helper functions -----------------------
// Basic collision detection and process
void SPHBase::check_collision(const std::shared_ptr<Particle>& p)
{
    // coefficient of restitution, you can make this parameter adjustable in the UI 
    double restitution = 0.2; 

    // add epsilon offset to avoid particles sticking to the boundary
    Vector3d eps_ = 0.0001 * (box_max_ - box_min_);

    for (int i = 0; i < 3; i++) {
        if (p->x()[i] < box_min_[i]) {
            p->x()[i] = box_min_[i] + eps_[i];
            p->vel()[i] = -restitution * p->vel()[i];
        }
        if (p->x()[i] > box_max_[i]) {
            p->x()[i] = box_max_[i] - eps_[i];
            p->vel()[i] = -restitution * p->vel()[i];
        }
    }
}

// For display
MatrixXd SPHBase::get_vel_color_jet()
{
    MatrixXd vel_color = MatrixXd::Zero(vel_.rows(), 3);
    double max_vel_norm = vel_.rowwise().norm().maxCoeff();
    double min_vel_norm = vel_.rowwise().norm().minCoeff();

    auto c = colormap_jet;

    for (int i = 0; i < vel_.rows(); i++) {
        double vel_norm = vel_.row(i).norm();
        int idx = 0;
        if (fabs(max_vel_norm - min_vel_norm) > 1e-6) {
            idx = static_cast<int>(
                floor((vel_norm - min_vel_norm) / (max_vel_norm - min_vel_norm) * 255));
        }
        vel_color.row(i) << c[idx][0], c[idx][1], c[idx][2];
    }
    return vel_color;
}

void SPHBase::reset()
{
    X_ = init_X_;
    vel_ = MatrixXd::Zero(X_.rows(), X_.cols());

    for (auto& p : ps_.particles()) {
        p->vel() = Vector3d::Zero();
        p->x() = init_X_.row(p->idx()).transpose();
    }
    diagnostics_ = Diagnostics{};
}

void SPHBase::update_diagnostics(int pressure_iterations, double pressure_residual)
{
    const auto& particles = ps_.particles();
    const int n_particles = static_cast<int>(particles.size());
    diagnostics_ = Diagnostics{};
    diagnostics_.particle_count = n_particles;
    diagnostics_.pressure_iterations = pressure_iterations;
    diagnostics_.pressure_residual = pressure_residual;

    if (n_particles == 0) {
        return;
    }

    const double density0 = ps_.density0();
    double min_density = std::numeric_limits<double>::max();
    double max_density = 0.0;
    double density_sum = 0.0;
    double relative_density_error_sum = 0.0;
    double max_velocity = 0.0;
    double kinetic_energy = 0.0;
    double neighbor_count_sum = 0.0;

    for (const auto& p : particles) {
        const double density = p->density();
        const double speed = p->vel().norm();

        min_density = std::min(min_density, density);
        max_density = std::max(max_density, density);
        density_sum += density;
        relative_density_error_sum += std::abs(density - density0) / density0;
        max_velocity = std::max(max_velocity, speed);
        kinetic_energy += 0.5 * ps_.mass() * p->vel().squaredNorm();
        neighbor_count_sum += static_cast<double>(p->neighbors().size());
    }

    diagnostics_.min_density = min_density;
    diagnostics_.max_density = max_density;
    diagnostics_.avg_density = density_sum / n_particles;
    diagnostics_.avg_relative_density_error =
        relative_density_error_sum / n_particles;
    diagnostics_.max_velocity = max_velocity;
    diagnostics_.kinetic_energy = kinetic_energy;
    diagnostics_.avg_neighbor_count = neighbor_count_sum / n_particles;

    if (enable_debug_output) {
        std::cout << "[SPH diagnostics] particles=" << diagnostics_.particle_count
                  << " avg_density=" << diagnostics_.avg_density
                  << " rel_density_error="
                  << diagnostics_.avg_relative_density_error
                  << " max_velocity=" << diagnostics_.max_velocity
                  << " avg_neighbors=" << diagnostics_.avg_neighbor_count
                  << " pressure_iters=" << diagnostics_.pressure_iterations
                  << " pressure_residual=" << diagnostics_.pressure_residual
                  << std::endl;
    }
}

// ---------------------------------------------------------------------------------------
}  // namespace USTC_CG::node_sph_fluid
