#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>
#include <chrono>
#include <set>
#include <iostream>
#include <map>
#include "utils.h"

#define TIC(name) auto start_##name = std::chrono::high_resolution_clock::now();
#define TOC(name)                                                           \
    auto end_##name = std::chrono::high_resolution_clock::now();            \
    if (enable_time_profiling)                                              \
        std::cout << "Time taken by " << #name << ": "                      \
                  << std::chrono::duration_cast<std::chrono::microseconds>( \
                         end_##name - start_##name)                         \
                         .count()                                           \
                  << " microseconds\n";

namespace USTC_CG::mass_spring {

using namespace Eigen;
using Edge = std::pair<int, int>;
using EdgeSet = std::set<Edge>;
using MatrixXd = Eigen::MatrixXd;
using SparseMatrix_d = Eigen::SparseMatrix<double>;
using Trip_d = Eigen::Triplet<double>;

class MassSpring {
   public:
    MassSpring() = default;

    virtual ~MassSpring() = default;

    enum TimeIntegrator { IMPLICIT_EULER = 0, SEMI_IMPLICIT_EULER = 1 };

    MassSpring(const Eigen::MatrixXd &X, const EdgeSet &E);

    // Advance the simulation by one time step using the selected integrator.
    virtual void step();
    void reset();

    // Spring energy and derivatives for E = 0.5 * k * sum((|x_i-x_j|-L)^2).
    virtual double computeEnergy(double stiffness);
    virtual Eigen::MatrixXd computeGrad(double stiffness);
    virtual Eigen::SparseMatrix<double> computeHessianSparse(double stiffness);

    // make matrix positive definite
    // Eigen::SparseMatrix<double> makeSPD(const Eigen::SparseMatrix<double>
    // &A);
    bool checkSPD(const Eigen::SparseMatrix<double> &A);

    Eigen::MatrixXd getVelocity() const
    {
        return vel;
    }
    Eigen::MatrixXd getX() const
    {
        return X;
    }
    Eigen::MatrixXd getInitX() const
    {
        return init_X;
    }
    const std::vector<bool>& getDirichletMask() const
    {
        return dirichlet_bc_mask;
    }

    // Linear penalty + normal damping force from a sphere collider. The
    // returned value is force, not acceleration; step() divides it by the
    // lumped vertex mass.
    Eigen::MatrixXd getSphereCollisionForce(
        Eigen::Vector3d center,
        double radius) const;

    // Dirichlet boundary helpers. The mask marks fixed simulated vertices.
    bool set_dirichlet_bc_mask(const std::vector<bool> &mask);
    bool set_edge_stiffness_scales(const std::map<Edge, double> &scales);
    bool update_dirichlet_bc_vertices(const MatrixXd &control_vertices);
    bool init_dirichlet_bc_vertices_control_pair(
        const MatrixXd &control_vertices,
        const std::vector<bool> &control_mask);

    // Simulation parameters configured by hw9_mass_spring node inputs.
    double stiffness = 1000.0;
    double damping = 0.995;
    enum TimeIntegrator time_integrator = IMPLICIT_EULER;
    double mass = 1.0;  // total mass of the mesh
    double h = 1e-2;    // time step
    int implicit_newton_iterations = 5;
    double implicit_delta_tolerance = 1e-7;
    int semi_implicit_substeps = 0;
    Eigen::Vector3d gravity = { 0, 0, -9.8 };
    Eigen::Vector3d wind_ext_acc = {
        0,
        0,
        0
    };  // (HW TODO) feel free to change the wind acceleration

    // Optional sphere-collision parameters.
    double collision_penalty_k = 10000.0;
    double collision_damping = 50.0;
    double collision_scale_factor = 1.1;
    Eigen::Vector3f sphere_center = Eigen::Vector3f(0, -0.5, 0.2);
    double sphere_radius = 0.4;

    // Feature switches configured by hw9_mass_spring node inputs.
    bool enable_sphere_collision = false;
    bool enable_time_profiling = false;
    bool enable_make_SPD = false;
    bool enable_check_SPD = false;
    bool enable_damping = true;
    bool enable_debug_output = false;

   protected:
    Eigen::MatrixXd init_X;  // For reset
    Eigen::MatrixXd X;
    Eigen::MatrixXd vel;
    EdgeSet E;
    std::vector<double> E_rest_length;
    std::vector<double> E_stiffness_scale;
    std::vector<bool> dirichlet_bc_mask;  // mask for marking fixed points
                                          // (Dirichlet boundary condition)
    std::vector<std::pair<int, int>> dirichlet_bc_control_pair;
};
}  // namespace USTC_CG::mass_spring
