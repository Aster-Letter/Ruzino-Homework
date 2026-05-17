#pragma once
#include <memory>
#include <vector>

#include "MassSpring.h"

namespace USTC_CG::mass_spring {
// Liu et al. 2013 fast mass-spring solver.
//
// The solver rewrites spring energy with auxiliary edge directions and
// alternates local projection and global linear solve:
//   A x = M y + h^2 J d, where A = M + h^2 L.
// A is constant while topology, mass, stiffness, and h stay unchanged, so it is
// assembled and prefactorized once at construction/reset.
class FastMassSpring : public MassSpring {
   public:
    FastMassSpring() = default;
    ~FastMassSpring() = default;

    FastMassSpring(
        const Eigen::MatrixXd& X,
        const EdgeSet& E,
        const float stiffness,
        const float h);

    // Advance one frame with the prefactorized Liu13 local-global solver.
    void step() override;

    // Number of local-global iterations per frame. More iterations approach
    // the implicit Newton result but increase frame cost.
    unsigned max_iter = 20;  // (HW Optional) expose this parameter in the node.

   protected:
    using LinearSolver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;

    // Rebuild and prefactorize A when stiffness, h, mass, topology, or boundary
    // constraints change.
    void rebuild_prefactorized_system();

    // Assemble A = M + h^2 L. Fixed vertices are constrained as absolute
    // positions in the global solve.
    Eigen::SparseMatrix<double> build_system_matrix() const;

    // Local step: project each current spring vector to its rest length.
    std::vector<Eigen::Vector3d> compute_edge_projections() const;

    // Predicted inertial position y = x + h v + h^2 M^{-1} f_ext.
    Eigen::MatrixXd compute_inertial_prediction() const;

    // Global step RHS: M y + h^2 J d, plus absolute values for fixed vertices.
    // Collision currently enters through y as an explicit linear penalty +
    // normal damping force. A stricter post-solve projection can still be added
    // later if force-only handling leaves visible penetration.
    Eigen::MatrixXd build_global_rhs(
        const Eigen::MatrixXd& y,
        const std::vector<Eigen::Vector3d>& edge_projections) const;

    // Solve the global linear system and return n x 3 positions.
    Eigen::MatrixXd solve_global_positions(const Eigen::MatrixXd& rhs) const;

    void enforce_fixed_vertices(Eigen::MatrixXd& positions) const;
    void update_velocity_from_positions(const Eigen::MatrixXd& old_positions);

    double mass_per_vertex() const;
    bool needs_refactorization() const;

    Eigen::SparseMatrix<double> system_matrix_;
    LinearSolver system_solver_;
    bool system_prefactorized_ = false;
    double cached_stiffness_ = -1.0;
    double cached_h_ = -1.0;
    double cached_mass_ = -1.0;
    std::vector<bool> cached_dirichlet_bc_mask_;
    std::vector<double> cached_edge_stiffness_scale_;
};
}  // namespace USTC_CG::mass_spring
