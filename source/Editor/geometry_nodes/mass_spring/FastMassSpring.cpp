#include "FastMassSpring.h"

#include <cmath>
#include <iostream>

namespace USTC_CG::mass_spring {
namespace {
constexpr double kLengthEps = 1e-10;

// The Liu13 solver stores all vertex coordinates as
// [x0, y0, z0, x1, y1, z1, ...]. A vertex is fixed if any of its three
// corresponding scalar degrees of freedom must be prescribed by Dirichlet BC.
bool is_fixed_vertex(const std::vector<bool>& mask, int vertex_id)
{
    return vertex_id >= 0 && vertex_id < static_cast<int>(mask.size()) &&
           mask[vertex_id];
}

bool is_fixed_dof(const std::vector<bool>& mask, int dof_id)
{
    return is_fixed_vertex(mask, dof_id / 3);
}

void add_block(
    std::vector<Trip_d>& triplets,
    int row_vertex,
    int col_vertex,
    const Eigen::Matrix3d& block)
{
    // Each graph edge contributes a 3x3 block because every spring couples
    // the x/y/z coordinates of two endpoint vertices.
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            triplets.emplace_back(
                3 * row_vertex + r, 3 * col_vertex + c, block(r, c));
        }
    }
}
}  // namespace

FastMassSpring::FastMassSpring(
    const Eigen::MatrixXd& X,
    const EdgeSet& E,
    const float stiffness,
    const float h)
    : MassSpring(X, E)
{
    // construct L and J at initialization
    std::cout << "init fast mass spring" << std::endl;

    this->stiffness = stiffness;
    this->h = h;
    rebuild_prefactorized_system();
}

void FastMassSpring::step()
{
    TIC(fast_mass_spring_step)

    // A = M + h^2 L depends only on topology, mass, stiffness, h, and fixed
    // vertices. Reuse the factorization while those inputs stay unchanged.
    if (needs_refactorization()) {
        rebuild_prefactorized_system();
    }
    if (!system_prefactorized_) {
        std::cerr << "FastMassSpring: prefactorization is not available."
                  << std::endl;
        return;
    }

    const Eigen::MatrixXd old_X = X;
    const Eigen::MatrixXd y = compute_inertial_prediction();

    // Local-global iteration:
    // 1. Local: for every edge, choose the closest rest-length spring vector d.
    // 2. Global: with d fixed, solve the quadratic system for vertex positions.
    // Repeating this reduces the surrogate energy and approaches the implicit
    // solution as max_iter increases.
    for (unsigned iter = 0; iter < max_iter; iter++) {
        const auto edge_projections = compute_edge_projections();
        const Eigen::MatrixXd rhs = build_global_rhs(y, edge_projections);
        X = solve_global_positions(rhs);
        enforce_fixed_vertices(X);
    }

    update_velocity_from_positions(old_X);
    TOC(fast_mass_spring_step)
}

void FastMassSpring::rebuild_prefactorized_system()
{
    // This is the main performance trick of Liu13: factorize the constant
    // system matrix once, then reuse it for all frames and local-global
    // iterations until a cached parameter changes.
    system_matrix_ = build_system_matrix();
    system_solver_.compute(system_matrix_);
    system_prefactorized_ = system_solver_.info() == Eigen::Success;

    cached_stiffness_ = stiffness;
    cached_h_ = h;
    cached_mass_ = mass;
    cached_dirichlet_bc_mask_ = dirichlet_bc_mask;

    if (!system_prefactorized_) {
        std::cerr << "FastMassSpring: failed to prefactorize system matrix."
                  << std::endl;
    }
}

Eigen::SparseMatrix<double> FastMassSpring::build_system_matrix() const
{
    const int n_vertices = static_cast<int>(X.rows());
    const double vertex_mass = mass_per_vertex();
    std::vector<Trip_d> triplets;
    triplets.reserve(n_vertices * 3 + E.size() * 36);

    // Mass part: lumped mass matrix M. Each vertex has the same scalar mass on
    // x/y/z, so the block is vertex_mass * I_3.
    for (int vertex = 0; vertex < n_vertices; ++vertex) {
        for (int d = 0; d < 3; ++d) {
            const int dof = 3 * vertex + d;
            triplets.emplace_back(dof, dof, vertex_mass);
        }
    }

    const Eigen::Matrix3d edge_block =
        (h * h * stiffness) * Eigen::Matrix3d::Identity();
    // Laplacian part: each spring contributes k * [[I, -I], [-I, I]].
    // The h^2 factor comes from the implicit time integration objective.
    for (const auto& edge : E) {
        add_block(triplets, edge.first, edge.first, edge_block);
        add_block(triplets, edge.second, edge.second, edge_block);
        add_block(triplets, edge.first, edge.second, -edge_block);
        add_block(triplets, edge.second, edge.first, -edge_block);
    }

    std::vector<Trip_d> constrained_triplets;
    constrained_triplets.reserve(triplets.size() + dirichlet_bc_mask.size() * 3);
    // Absolute-position Dirichlet constraints: remove rows/columns that couple
    // to fixed coordinates, then insert identity rows for those coordinates.
    // build_global_rhs() compensates free rows for the removed fixed columns.
    for (const auto& triplet : triplets) {
        if (!is_fixed_dof(dirichlet_bc_mask, triplet.row()) &&
            !is_fixed_dof(dirichlet_bc_mask, triplet.col())) {
            constrained_triplets.push_back(triplet);
        }
    }
    for (int vertex = 0; vertex < static_cast<int>(dirichlet_bc_mask.size());
         ++vertex) {
        if (!dirichlet_bc_mask[vertex]) {
            continue;
        }
        for (int d = 0; d < 3; ++d) {
            const int dof = 3 * vertex + d;
            constrained_triplets.emplace_back(dof, dof, 1.0);
        }
    }

    Eigen::SparseMatrix<double> A(n_vertices * 3, n_vertices * 3);
    A.setFromTriplets(constrained_triplets.begin(), constrained_triplets.end());
    A.makeCompressed();
    return A;
}

std::vector<Eigen::Vector3d> FastMassSpring::compute_edge_projections() const
{
    std::vector<Eigen::Vector3d> projections;
    projections.reserve(E.size());

    // Local step from Liu13:
    //   d_i = L_i * (x_a - x_b) / ||x_a - x_b||
    // This is the closest vector with rest length L_i to the current spring
    // vector, and is independent for every edge.
    unsigned edge_id = 0;
    for (const auto& edge : E) {
        Eigen::Vector3d direction = X.row(edge.first) - X.row(edge.second);
        const double length = direction.norm();

        if (length > kLengthEps) {
            direction *= E_rest_length[edge_id] / length;
        }
        else {
            direction.setZero();
        }

        projections.push_back(direction);
        ++edge_id;
    }
    return projections;
}

Eigen::MatrixXd FastMassSpring::compute_inertial_prediction() const
{
    // y is the position predicted by current velocity and explicit external
    // acceleration. It is fixed during all local-global iterations of this
    // frame, as required by the implicit Euler objective.
    Eigen::MatrixXd y = X + h * vel;
    y.rowwise() += (h * h * (gravity + wind_ext_acc)).transpose();

    if (enable_sphere_collision) {
        // Collision is evaluated explicitly from the current position/velocity.
        // The force combines linear penalty and inward normal damping.
        y += (h * h / mass_per_vertex()) *
             getSphereCollisionForce(sphere_center.cast<double>(), sphere_radius);
    }

    return y;
}

Eigen::MatrixXd FastMassSpring::build_global_rhs(
    const Eigen::MatrixXd& y,
    const std::vector<Eigen::Vector3d>& edge_projections) const
{
    // Global step right-hand side:
    //   b = M y + h^2 J d.
    // M y pulls the solution toward inertial motion, and J d pulls edge
    // differences toward the projected rest-length directions from local step.
    Eigen::MatrixXd rhs = mass_per_vertex() * y;

    unsigned edge_id = 0;
    for (const auto& edge : E) {
        const Eigen::Vector3d contribution =
            h * h * stiffness * edge_projections[edge_id];
        rhs.row(edge.first) += contribution.transpose();
        rhs.row(edge.second) -= contribution.transpose();
        ++edge_id;
    }

    // Fixed-position constraints are enforced by removing fixed columns from A.
    // Compensate the free RHS by subtracting the omitted A_free,fixed * x_fixed
    // terms, where each spring contributes -h^2*k*I to the off-diagonal block.
    const double edge_coeff = h * h * stiffness;
    for (const auto& edge : E) {
        const bool first_fixed =
            is_fixed_vertex(dirichlet_bc_mask, edge.first);
        const bool second_fixed =
            is_fixed_vertex(dirichlet_bc_mask, edge.second);

        if (first_fixed && !second_fixed) {
            rhs.row(edge.second) += edge_coeff * init_X.row(edge.first);
        }
        else if (!first_fixed && second_fixed) {
            rhs.row(edge.first) += edge_coeff * init_X.row(edge.second);
        }
    }

    for (int vertex = 0; vertex < static_cast<int>(dirichlet_bc_mask.size());
         ++vertex) {
        if (dirichlet_bc_mask[vertex]) {
            rhs.row(vertex) = init_X.row(vertex);
        }
    }

    return flatten(rhs);
}

Eigen::MatrixXd FastMassSpring::solve_global_positions(
    const Eigen::MatrixXd& rhs) const
{
    // The solver expects a flattened 3n x 1 vector and returns the same layout.
    // unflatten() converts it back to the n x 3 vertex matrix used elsewhere.
    Eigen::MatrixXd x_flat = system_solver_.solve(rhs);
    if (system_solver_.info() != Eigen::Success) {
        std::cerr << "FastMassSpring: global solve failed." << std::endl;
        return X;
    }

    return unflatten(x_flat);
}

void FastMassSpring::enforce_fixed_vertices(Eigen::MatrixXd& positions) const
{
    for (int vertex = 0; vertex < positions.rows(); ++vertex) {
        if (is_fixed_vertex(dirichlet_bc_mask, vertex)) {
            positions.row(vertex) = init_X.row(vertex);
        }
    }
}

void FastMassSpring::update_velocity_from_positions(
    const Eigen::MatrixXd& old_positions)
{
    // After an implicit-style position solve, velocity is reconstructed from
    // displacement over the time step instead of being integrated separately.
    vel = (X - old_positions) / h;
    for (int vertex = 0; vertex < vel.rows(); ++vertex) {
        if (is_fixed_vertex(dirichlet_bc_mask, vertex)) {
            vel.row(vertex).setZero();
        }
    }
}

double FastMassSpring::mass_per_vertex() const
{
    return X.rows() > 0 ? mass / static_cast<double>(X.rows()) : 0.0;
}

bool FastMassSpring::needs_refactorization() const
{
    return !system_prefactorized_ || cached_stiffness_ != stiffness ||
           cached_h_ != h || cached_mass_ != mass ||
           cached_dirichlet_bc_mask_ != dirichlet_bc_mask;
}

}  // namespace USTC_CG::mass_spring
