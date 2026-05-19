#pragma once 
#include <Eigen/Dense>
#include "particle_system.h"
#include <memory>
#include <chrono>
#include <cstddef>

namespace USTC_CG::sph_fluid {
#define TIC(name) auto start_##name = std::chrono::high_resolution_clock::now();
#define TOC(name)                                                           \
    auto end_##name = std::chrono::high_resolution_clock::now();            \
    if (enable_time_profiling)                                              \
        std::cout << "Time taken by " << #name << ": "                      \
                  << std::chrono::duration_cast<std::chrono::microseconds>( \
                         end_##name - start_##name)                         \
                         .count()                                           \
                  << " microseconds\n";

class SPHBase {
   public:
    struct Diagnostics {
        int particle_count = 0;
        double min_density = 0.0;
        double max_density = 0.0;
        double avg_density = 0.0;
        double avg_relative_density_error = 0.0;
        double max_velocity = 0.0;
        double kinetic_energy = 0.0;
        double avg_neighbor_count = 0.0;
        int pressure_iterations = 0;
        double pressure_residual = 0.0;
    };

    SPHBase() = default;
    SPHBase(const Eigen::MatrixXd& X, const Vector3d& box_min, const Vector3d& box_max);
    virtual ~SPHBase() = default;

    virtual void step();
    virtual void reset();

    inline Eigen::MatrixXd getX() const
    {
        return X_;
    };
    inline Eigen::MatrixXd getVel() const
    {
        return vel_;
    };

    // SPH kernel function: h is the support radius, instead of time step size 
    static double W(const Eigen::Vector3d& r, double h);
    static Eigen::Vector3d grad_W(const Eigen::Vector3d& r, double h);
    static double W_zero(double h);

    // SPH functions
    virtual void compute_density();

    virtual Vector3d compute_viscosity_acceleration(
        const std::shared_ptr<Particle>& p,
        const std::shared_ptr<Particle>& q);

    virtual void compute_pressure_gradient_acceleration();

    virtual void compute_non_pressure_acceleration();

    virtual void compute_pressure();

    virtual void check_collision(const std::shared_ptr<Particle>& p);

    virtual void advect();

    ParticleSystem& ps()
    {
        return ps_;
    }
    double& dt()
    {
        return dt_;
    }
    double& viscosity()
    {
        return viscosity_;
    }
    Vector3d& gravity()
	{
		return gravity_;
	}

    Vector3d gravity_ = Vector3d(0, 0, -9.8);

    // useful switches
    bool enable_debug_output = false;
    bool enable_time_profiling = false;

    // for display: generate color for each particle based on its velocity
    MatrixXd get_vel_color_jet(); 

    const Diagnostics& diagnostics() const
    {
        return diagnostics_;
    }
   
  protected:
    void update_diagnostics(int pressure_iterations = 0, double pressure_residual = 0.0);

    ParticleSystem ps_;
    double dt_ = 0.005;  // You can adjust this parameter in the UI of node "SPH Fluid"
    double viscosity_ = 0.03; // You can adjust this parameter in the UI of node "SPH Fluid"

    Vector3d box_min_, box_max_; // simulation box area

    Eigen::MatrixXd init_X_;
    Eigen::MatrixXd X_;
    Eigen::MatrixXd vel_;
    Diagnostics diagnostics_;
};
}  // namespace USTC_CG::node_sph_fluid
