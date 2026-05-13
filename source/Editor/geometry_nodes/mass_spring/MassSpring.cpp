#include "MassSpring.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace USTC_CG::mass_spring {
namespace {
constexpr double kLengthEps = 1e-10;
constexpr double kTripletEps = 1e-12;

double vertex_mass(double total_mass, int n_vertices)
{
    return n_vertices > 0 ? total_mass / static_cast<double>(n_vertices) : 0.0;
}

bool is_fixed_vertex(const std::vector<bool>& mask, int vertex_id)
{
    return vertex_id >= 0 && vertex_id < static_cast<int>(mask.size()) &&
           mask[vertex_id];
}

bool is_fixed_dof(const std::vector<bool>& mask, int dof_id)
{
    return is_fixed_vertex(mask, dof_id / 3);
}

void add_sparse_block(
    std::vector<Trip_d>& triplets,
    int row_vertex,
    int col_vertex,
    const Eigen::Matrix3d& block)
{
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            const double value = block(r, c);
            if (std::abs(value) > kTripletEps) {
                triplets.emplace_back(
                    3 * row_vertex + r, 3 * col_vertex + c, value);
            }
        }
    }
}

Eigen::SparseMatrix<double> add_lumped_mass_term(
    const Eigen::SparseMatrix<double>& elastic_hessian,
    double mass_per_vertex,
    double h)
{
    // Implicit Euler minimizes the increment potential. Its Hessian is
    // M / h^2 + d^2E/dx^2, where M is the lumped mass matrix.
    Eigen::SparseMatrix<double> A = elastic_hessian;
    const double mass_coeff = mass_per_vertex / (h * h);
    std::vector<Trip_d> mass_triplets;
    mass_triplets.reserve(A.rows());
    for (int i = 0; i < A.rows(); ++i) {
        mass_triplets.emplace_back(i, i, mass_coeff);
    }

    Eigen::SparseMatrix<double> mass_term(A.rows(), A.cols());
    mass_term.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
    A += mass_term;
    A.makeCompressed();
    return A;
}

Eigen::SparseMatrix<double> apply_dirichlet_delta_constraints(
    const Eigen::SparseMatrix<double>& A,
    const std::vector<bool>& fixed_mask)
{
    // The implicit branch solves for delta_x, not absolute x. Fixed vertices
    // therefore mean delta_x = 0. Removing both rows and columns keeps the
    // matrix symmetric for SimplicialLDLT, then identity rows prescribe zero
    // movement for the fixed scalar coordinates.
    std::vector<Trip_d> triplets;
    triplets.reserve(A.nonZeros() + fixed_mask.size() * 3);

    for (int col = 0; col < A.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, col); it; ++it) {
            const int row = it.row();
            const int column = it.col();
            if (!is_fixed_dof(fixed_mask, row) &&
                !is_fixed_dof(fixed_mask, column)) {
                triplets.emplace_back(row, column, it.value());
            }
        }
    }

    for (int vertex = 0; vertex < static_cast<int>(fixed_mask.size());
         ++vertex) {
        if (!fixed_mask[vertex]) {
            continue;
        }
        for (int d = 0; d < 3; ++d) {
            const int idx = 3 * vertex + d;
            triplets.emplace_back(idx, idx, 1.0);
        }
    }

    Eigen::SparseMatrix<double> constrained(A.rows(), A.cols());
    constrained.setFromTriplets(triplets.begin(), triplets.end());
    constrained.makeCompressed();
    return constrained;
}

void zero_fixed_rows(Eigen::MatrixXd& values, const std::vector<bool>& mask)
{
    for (int i = 0; i < values.rows(); ++i) {
        if (is_fixed_vertex(mask, i)) {
            values.row(i).setZero();
        }
    }
}

void restore_fixed_vertices(
    Eigen::MatrixXd& X,
    Eigen::MatrixXd& vel,
    const Eigen::MatrixXd& init_X,
    const std::vector<bool>& mask)
{
    for (int i = 0; i < X.rows(); ++i) {
        if (is_fixed_vertex(mask, i)) {
            X.row(i) = init_X.row(i);
            vel.row(i).setZero();
        }
    }
}
}  // namespace

MassSpring::MassSpring(const Eigen::MatrixXd& X, const EdgeSet& E)
{
    this->X = this->init_X = X;
    this->vel = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    this->E = E;

    std::cout << "number of edges: " << E.size() << std::endl;
    std::cout << "init mass spring" << std::endl;

    // Compute the rest pose edge length
    for (const auto& e : E) {
        Eigen::Vector3d x0 = X.row(e.first);
        Eigen::Vector3d x1 = X.row(e.second);
        this->E_rest_length.push_back((x0 - x1).norm());
    }

    // Initialize the mask for Dirichlet boundary condition
    dirichlet_bc_mask.resize(X.rows(), false);

    // (HW_TODO) Fix two vertices, feel free to modify this
    unsigned n_fix = sqrt(X.rows());  // Here we assume the cloth is square
    dirichlet_bc_mask[0] = true;
    dirichlet_bc_mask[n_fix - 1] = true;
}

void MassSpring::step()
{
    Eigen::Vector3d acceleration_ext = gravity + wind_ext_acc;
    const int n_vertices = static_cast<int>(X.rows());

    // The reason to not use 1.0 as mass per vertex: the cloth gets heavier as
    // we increase the resolution
    const double mass_per_vertex = vertex_mass(mass, n_vertices);
    if (mass_per_vertex <= 0.0 || h <= 0.0) {
        std::cerr << "Mass Spring: invalid mass or time step." << std::endl;
        return;
    }

    Eigen::MatrixXd collision_force =
        Eigen::MatrixXd::Zero(X.rows(), X.cols());
    if (enable_sphere_collision) {
        collision_force =
            getSphereCollisionForce(sphere_center.cast<double>(), sphere_radius);
    }

    if (time_integrator == IMPLICIT_EULER) {
        TIC(step)

        // One Newton iteration for the implicit Euler increment potential:
        //   g(x) = 1/(2h^2) (x-y)^T M (x-y) + E(x).
        // We start at current X, solve A * delta = -grad_g, and set
        // X_new = X_old + delta.
        Eigen::MatrixXd X_old = X;
        Eigen::MatrixXd y = X_old + h * vel;
        y.rowwise() += (h * h * acceleration_ext).transpose();
        y += (h * h / mass_per_vertex) * collision_force;

        Eigen::SparseMatrix<double> A =
            add_lumped_mass_term(computeHessianSparse(stiffness),
                                 mass_per_vertex,
                                 h);

        Eigen::MatrixXd grad_g = computeGrad(stiffness);
        grad_g += (mass_per_vertex / (h * h)) * (X - y);
        zero_fixed_rows(grad_g, dirichlet_bc_mask);

        Eigen::MatrixXd rhs = -flatten(grad_g);
        for (int i = 0; i < static_cast<int>(dirichlet_bc_mask.size()); ++i) {
            if (!dirichlet_bc_mask[i]) {
                continue;
            }
            for (int d = 0; d < 3; ++d) {
                rhs(3 * i + d, 0) = 0.0;
            }
        }

        A = apply_dirichlet_delta_constraints(A, dirichlet_bc_mask);

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed in implicit Euler."
                      << std::endl;
            return;
        }

        Eigen::MatrixXd delta_flat = solver.solve(rhs);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Linear solve failed in implicit Euler." << std::endl;
            return;
        }

        X = X_old + unflatten(delta_flat);
        restore_fixed_vertices(X, vel, init_X, dirichlet_bc_mask);
        vel = (X - X_old) / h;
        zero_fixed_rows(vel, dirichlet_bc_mask);

        TOC(step)
    }
    else if (time_integrator == SEMI_IMPLICIT_EULER) {
        // Semi-implicit Euler
        // Force is -grad(E). We divide by the lumped vertex mass to obtain
        // acceleration, then update velocity before position.
        Eigen::MatrixXd acceleration =
            -computeGrad(stiffness) / mass_per_vertex;
        acceleration.rowwise() += acceleration_ext.transpose();

        acceleration += collision_force / mass_per_vertex;
        zero_fixed_rows(acceleration, dirichlet_bc_mask);

        vel += acceleration * h;
        if (enable_damping) {
            vel *= damping;
        }
        zero_fixed_rows(vel, dirichlet_bc_mask);
        X += vel * h;
        restore_fixed_vertices(X, vel, init_X, dirichlet_bc_mask);
    }
    else {
        std::cerr << "Unknown time integrator!" << std::endl;
        return;
    }
}

// There are different types of mass spring energy:
// For this homework we will adopt Prof. Huamin Wang's energy definition
// introduced in GAMES103 course Lecture 2 E = 0.5 * stiffness * sum_{i=1}^{n}
// (||x_i - x_j|| - l)^2 There exist other types of energy definition, e.g.,
// Prof. Minchen Li's energy definition
// https://www.cs.cmu.edu/~15769-f23/lec/3_Mass_Spring_Systems.pdf
double MassSpring::computeEnergy(double stiffness)
{
    double sum = 0.;
    unsigned i = 0;
    for (const auto& e : E) {
        auto diff = X.row(e.first) - X.row(e.second);
        auto l = E_rest_length[i];
        sum += 0.5 * stiffness * std::pow((diff.norm() - l), 2);
        i++;
    }
    return sum;
}

// The gradient is the first derivative of energy, which is also the negative force
Eigen::MatrixXd MassSpring::computeGrad(double stiffness)
{
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    unsigned i = 0;
    for (const auto& e : E) {
        // For one spring e=(a,b):
        //   E_e = 0.5*k*(||x_a-x_b|| - L)^2
        //   dE/dx_a = k*(len-L)*(x_a-x_b)/len
        //   dE/dx_b = -dE/dx_a
        const int a = e.first;
        const int b = e.second;
        Eigen::Vector3d d = X.row(a) - X.row(b);
        const double L = E_rest_length[i];
        const double len = d.norm();

        if (len > kLengthEps) {
            Eigen::Vector3d grad = stiffness * (len - L) * d / len;
            g.row(a) += grad;
            g.row(b) -= grad;
        }
        i++;
    }
    return g;
}

Eigen::SparseMatrix<double> MassSpring::computeHessianSparse(double stiffness)
{
    unsigned n_vertices = X.rows();
    Eigen::SparseMatrix<double> H(n_vertices * 3, n_vertices * 3);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(E.size() * 36);  // Each edge contributes at most 36 non-zero entries

    unsigned i = 0;
    auto k = stiffness;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    for (const auto& e : E) {
        const int a = e.first;
        const int b = e.second;

        Eigen::Vector3d d = X.row(a) - X.row(b);
        const double L = E_rest_length[i];
        const double len = d.norm();

        if (len > kLengthEps) {
            Eigen::Matrix3d outer = d * d.transpose() / (len * len);
            Eigen::Matrix3d H_local;

            // The full spring Hessian can be indefinite in compression.
            // Following the homework hint, keep only the axial SPD part there.
            // This stabilizes the LDLT solve without changing the gradient.
            if (L > len) {
                H_local = k * outer;
            }
            else {
                H_local = k * ((1 - L / len) * (I - outer) + outer);
            }

            add_sparse_block(triplets, a, a, H_local);
            add_sparse_block(triplets, a, b, -H_local);
            add_sparse_block(triplets, b, a, -H_local);
            add_sparse_block(triplets, b, b, H_local);
        }

        i++;
    }

    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();
    return H;
}

bool MassSpring::checkSPD(const Eigen::SparseMatrix<double>& A)
{
    // Eigen::SimplicialLDLT<SparseMatrix_d> ldlt(A);
    // return ldlt.info() == Eigen::Success;
    Eigen::MatrixXd dense_A(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense_A);
    auto eigen_values = es.eigenvalues();
    return eigen_values.minCoeff() >= 1e-10;
}

void MassSpring::reset()
{
    std::cout << "reset" << std::endl;
    this->X = this->init_X;
    this->vel.setZero();
}

// ----------------------------------------------------------------------------------
// (HW Optional) Bonus part
Eigen::MatrixXd MassSpring::getSphereCollisionForce(
    Eigen::Vector3d center,
    double radius) const
{
    Eigen::MatrixXd force = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); i++) {
        if (is_fixed_vertex(dirichlet_bc_mask, i)) {
            continue;
        }

        Eigen::Vector3d offset = X.row(i).transpose() - center;
        const double distance = offset.norm();
        const double active_radius = collision_scale_factor * radius;
        const double penetration = active_radius - distance;

        // Linear penalty + normal damping collision model:
        //   f = k_penalty * max(scale*r - distance, 0) * outward_normal
        //       - damping * min(v dot outward_normal, 0) * outward_normal.
        // The damping term only opposes inward normal velocity, so it reduces
        // repeated tunneling/oscillation without pulling separating vertices
        // back toward the collider.
        if (penetration <= 0.0) {
            continue;
        }

        Eigen::Vector3d normal(0.0, 0.0, 1.0);
        if (distance > kLengthEps) {
            normal = offset / distance;
        }

        const double inward_normal_speed = std::min(vel.row(i).dot(normal), 0.0);
        const Eigen::Vector3d penalty_force =
            collision_penalty_k * penetration * normal;
        const Eigen::Vector3d damping_force =
            -collision_damping * inward_normal_speed * normal;
        force.row(i) = (penalty_force + damping_force).transpose();
    }
    return force;
}
// ----------------------------------------------------------------------------------

bool MassSpring::set_dirichlet_bc_mask(const std::vector<bool>& mask)
{
    if (mask.size() == X.rows()) {
        dirichlet_bc_mask = mask;
        return true;
    }
    else
        return false;
}

bool MassSpring::update_dirichlet_bc_vertices(const MatrixXd& control_vertices)
{
    for (int i = 0; i < dirichlet_bc_control_pair.size(); i++) {
        int idx = dirichlet_bc_control_pair[i].first;
        int control_idx = dirichlet_bc_control_pair[i].second;
        X.row(idx) = control_vertices.row(control_idx);
    }

    return true;
}

bool MassSpring::init_dirichlet_bc_vertices_control_pair(
    const MatrixXd& control_vertices,
    const std::vector<bool>& control_mask)
{
    if (control_mask.size() != control_vertices.rows())
        return false;

    // TODO: optimize this part from O(n) to O(1)
    // First, get selected_control_vertices
    std::vector<VectorXd> selected_control_vertices;
    std::vector<int> selected_control_idx;
    for (int i = 0; i < control_mask.size(); i++) {
        if (control_mask[i]) {
            selected_control_vertices.push_back(control_vertices.row(i));
            selected_control_idx.push_back(i);
        }
    }

    // Then update mass spring fixed vertices
    for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
        if (dirichlet_bc_mask[i]) {
            // O(n^2) nearest point search, can be optimized
            // -----------------------------------------
            int nearest_idx = 0;
            double nearst_dist = 1e6;
            VectorXd X_i = X.row(i);
            for (int j = 0; j < selected_control_vertices.size(); j++) {
                double dist = (X_i - selected_control_vertices[j]).norm();
                if (dist < nearst_dist) {
                    nearst_dist = dist;
                    nearest_idx = j;
                }
            }
            //-----------------------------------------

            X.row(i) = selected_control_vertices[nearest_idx];
            dirichlet_bc_control_pair.push_back(
                std::make_pair(i, selected_control_idx[nearest_idx]));
        }
    }

    return true;
}

}  // namespace USTC_CG::mass_spring
