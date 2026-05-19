#include "wcsph.h"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace Eigen;

namespace USTC_CG::sph_fluid {

namespace {
constexpr int kParallelParticleThreshold = 256;
}

WCSPH::WCSPH(const MatrixXd& X, const Vector3d& box_min, const Vector3d& box_max)
    : SPHBase(X, box_min, box_max)
{
}

void WCSPH::compute_density()
{
    auto& particles = ps_.particles();
    const double mass = ps_.mass();
    const double h = ps_.h();
    const double w0 = W_zero(h);
    const double density0 = ps_.density0();
    const int n_particles = static_cast<int>(particles.size());

#pragma omp parallel for if(n_particles > kParallelParticleThreshold) schedule(static)
    for (int i = 0; i < n_particles; i++) {
        const auto& p = particles[i];
        double rho = mass * w0;

        for (const auto& q : p->neighbors()) {
            rho += mass * W(p->x() - q->x(), h);
        }

        p->density() = rho;

        const double pressure =
            stiffness_ * (std::pow(rho / density0, exponent_) - 1.0);
        p->pressure() = std::max(0.0, pressure);
    }
}

void WCSPH::step()
{
    TIC(step)

    ps_.assign_particles_to_cells();
    ps_.search_neighbors();

    compute_density();
    compute_non_pressure_acceleration();
    compute_pressure_gradient_acceleration();
    advect();
    update_diagnostics();

    TOC(step)
}
}  // namespace USTC_CG::node_sph_fluid
