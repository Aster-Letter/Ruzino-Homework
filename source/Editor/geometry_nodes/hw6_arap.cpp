#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <algorithm>
#include <array>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"
#include "geom_node_base.h"

namespace {
using MeshType = OpenMesh::PolyMesh_ArrayKernelT<>;

enum class ConstraintMode {
    SoftPenalty = 0,
    HardElimination = 1,
};

enum class ParameterizationAlgorithm {
    ARAP = 0,
    ASAP = 1,
    Hybrid = 2,
};

struct ArapOptions {
    int max_iterations = 10;
    ParameterizationAlgorithm algorithm = ParameterizationAlgorithm::ARAP;
    ConstraintMode constraint_mode = ConstraintMode::SoftPenalty;
    double soft_penalty_weight = 1e8;
    double hybrid_lambda = 0.0;
};

struct ArapConstraints {
    std::vector<int> fixed_vertex_ids;
    std::vector<Eigen::Vector2d> fixed_positions;
};

struct FacePrecomputation {
    std::array<int, 3> vertex_ids{};
    std::array<Eigen::Vector2d, 3> rest_positions{};
    std::array<double, 3> cotangent_weights{};
    bool is_valid = false;
    std::array<Eigen::Vector2d, 3> opposite_edges{};
    double doubled_area = 0.0;
};

struct UvQualityStats {
    int flipped_face_count = 0;
    int degenerate_face_count = 0;
};

struct HybridLocalConstants {
    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;
};

struct ConstraintIndexMaps {
    std::vector<char> is_fixed;
    std::vector<int> fixed_slot;
};

struct FreeVertexMapping {
    std::vector<int> free_vertex_ids;
    std::vector<int> full_to_free;
};

struct ArapPrecomputation {
    int num_vertices = 0;
    int num_faces = 0;
    std::vector<FacePrecomputation> face_data;
    Eigen::SparseMatrix<double> laplacian_matrix;
    Eigen::SparseMatrix<double> system_matrix;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    std::vector<int> free_vertex_ids;
    std::vector<int> full_to_free;
};

struct ArapState {
    std::vector<glm::vec2> current_uv;
    std::vector<Eigen::Matrix2d> local_transforms;
    UvQualityStats quality_stats;
};

using TriangleSignature = std::array<int, 3>;

bool collect_triangle_vertices(
    const MeshType& mesh,
    const MeshType::FaceHandle& face,
    std::array<int, 3>& vertex_ids);

ConstraintMode parse_constraint_mode(int raw_mode)
{
    return raw_mode == 1 ? ConstraintMode::HardElimination
                         : ConstraintMode::SoftPenalty;
}

ParameterizationAlgorithm parse_parameterization_algorithm(int raw_algorithm)
{
    switch (std::clamp(raw_algorithm, 0, 2)) {
    case 1:
        return ParameterizationAlgorithm::ASAP;
    case 2:
        return ParameterizationAlgorithm::Hybrid;
    case 0:
    default:
        return ParameterizationAlgorithm::ARAP;
    }
}

Eigen::Matrix2d make_similarity_transform(double a, double b)
{
    Eigen::Matrix2d transform;
    transform << a, b,
        -b, a;
    return transform;
}

std::vector<double> solve_depressed_cubic_real_roots(
    double cubic_coeff,
    double linear_coeff,
    double constant_coeff)
{
    constexpr double kEps = 1e-12;
    std::vector<double> roots;

    if (std::abs(cubic_coeff) <= kEps) {
        if (std::abs(linear_coeff) <= kEps) {
            return roots;
        }

        roots.push_back(-constant_coeff / linear_coeff);
        return roots;
    }

    const double p = linear_coeff / cubic_coeff;
    const double q = constant_coeff / cubic_coeff;
    const double discriminant =
        0.25 * q * q + (p * p * p) / 27.0;

    if (discriminant > kEps) {
        const double sqrt_discriminant = std::sqrt(discriminant);
        const double u = std::cbrt(-0.5 * q + sqrt_discriminant);
        const double v = std::cbrt(-0.5 * q - sqrt_discriminant);
        roots.push_back(u + v);
        return roots;
    }

    if (std::abs(discriminant) <= kEps) {
        const double u = std::cbrt(-0.5 * q);
        roots.push_back(2.0 * u);
        roots.push_back(-u);
        return roots;
    }

    const double radius = 2.0 * std::sqrt(-p / 3.0);
    const double angle = std::acos(
        std::clamp(
            -0.5 * q / std::sqrt(-(p * p * p) / 27.0),
            -1.0,
            1.0));
    constexpr double kTwoPi = 6.28318530717958647692;
    for (int branch = 0; branch < 3; ++branch) {
        roots.push_back(radius * std::cos((angle + kTwoPi * branch) / 3.0));
    }

    return roots;
}

HybridLocalConstants compute_hybrid_local_constants(
    const FacePrecomputation& face_data,
    const std::vector<glm::vec2>& uv)
{
    static constexpr std::array<std::array<int, 2>, 3> kOppositeEdgeEndpoints = {
        std::array<int, 2>{1, 2},
        std::array<int, 2>{2, 0},
        std::array<int, 2>{0, 1},
    };

    HybridLocalConstants constants;
    for (int edge_index = 0; edge_index < 3; ++edge_index) {
        const double weight = face_data.cotangent_weights[edge_index];
        if (std::abs(weight) <= 1e-12) {
            continue;
        }

        const int from = face_data.vertex_ids[kOppositeEdgeEndpoints[edge_index][0]];
        const int to = face_data.vertex_ids[kOppositeEdgeEndpoints[edge_index][1]];
        const Eigen::Vector2d du(
            static_cast<double>(uv[from].x - uv[to].x),
            static_cast<double>(uv[from].y - uv[to].y));
        const Eigen::Vector2d& dv = face_data.opposite_edges[edge_index];

        constants.c1 += weight * dv.squaredNorm();
        constants.c2 += weight * du.dot(dv);
        constants.c3 += weight * (du.x() * dv.y() - du.y() * dv.x());
    }

    return constants;
}

double evaluate_hybrid_local_energy(
    const HybridLocalConstants& constants,
    double lambda,
    double a,
    double b)
{
    const double squared_scale = a * a + b * b;
    const double data_term =
        constants.c1 * squared_scale -
        2.0 * constants.c2 * a -
        2.0 * constants.c3 * b;
    const double penalty_term =
        std::max(0.0, lambda) * std::pow(squared_scale - 1.0, 2.0);
    return data_term + penalty_term;
}

Eigen::Matrix2d compute_hybrid_local_transform(
    const FacePrecomputation& face_data,
    const std::vector<glm::vec2>& uv,
    double lambda)
{
    const HybridLocalConstants constants =
        compute_hybrid_local_constants(face_data, uv);
    const double magnitude = std::hypot(constants.c2, constants.c3);
    if (magnitude <= 1e-12) {
        return Eigen::Matrix2d::Identity();
    }

    if (lambda <= 1e-12) {
        return make_similarity_transform(
            constants.c2 / constants.c1,
            constants.c3 / constants.c1);
    }

    const double cubic_coeff = 2.0 * lambda;
    const double linear_coeff = constants.c1 - 2.0 * lambda;
    const double constant_coeff = -magnitude;
    const std::vector<double> candidate_scales =
        solve_depressed_cubic_real_roots(cubic_coeff, linear_coeff, constant_coeff);

    double best_energy = std::numeric_limits<double>::infinity();
    Eigen::Matrix2d best_transform = Eigen::Matrix2d::Identity();
    for (const double scale : candidate_scales) {
        const double a = scale * constants.c2 / magnitude;
        const double b = scale * constants.c3 / magnitude;
        const double energy = evaluate_hybrid_local_energy(constants, lambda, a, b);
        if (energy < best_energy) {
            best_energy = energy;
            best_transform = make_similarity_transform(a, b);
        }
    }

    return best_transform;
}

ConstraintIndexMaps build_constraint_index_maps(
    const ArapConstraints& constraints,
    int num_vertices)
{
    ConstraintIndexMaps maps;
    maps.is_fixed.assign(static_cast<size_t>(num_vertices), 0);
    maps.fixed_slot.assign(num_vertices, -1);

    int compact_fixed_index = 0;
    for (const int vertex_id : constraints.fixed_vertex_ids) {
        if (vertex_id < 0 || vertex_id >= num_vertices) {
            continue;
        }
        if (maps.is_fixed[vertex_id]) {
            continue;
        }

        maps.is_fixed[vertex_id] = 1;
        maps.fixed_slot[vertex_id] = compact_fixed_index++;
    }

    return maps;
}

FreeVertexMapping build_free_vertex_mapping(
    const std::vector<char>& is_fixed,
    int num_vertices)
{
    FreeVertexMapping mapping;
    mapping.full_to_free.assign(num_vertices, -1);

    for (int vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
        if (is_fixed[vertex_id]) {
            continue;
        }

        mapping.full_to_free[vertex_id] =
            static_cast<int>(mapping.free_vertex_ids.size());
        mapping.free_vertex_ids.push_back(vertex_id);
    }

    return mapping;
}

TriangleSignature build_triangle_signature(
    const MeshType& mesh,
    const MeshType::FaceHandle& face)
{
    TriangleSignature signature{};
    if (!collect_triangle_vertices(mesh, face, signature)) {
        throw std::runtime_error("ARAP: Topology check expects a triangulated mesh.");
    }

    std::sort(signature.begin(), signature.end());
    return signature;
}

std::vector<TriangleSignature> collect_triangle_signatures(const MeshType& mesh)
{
    std::vector<TriangleSignature> signatures;
    signatures.reserve(mesh.n_faces());

    for (const auto& face_handle : mesh.faces()) {
        signatures.push_back(build_triangle_signature(mesh, face_handle));
    }

    std::sort(signatures.begin(), signatures.end());
    return signatures;
}

double penalty_weight_from_slider(int raw_slider_value)
{
    const int clamped_value = std::clamp(raw_slider_value, 2, 12);
    return std::pow(10.0, static_cast<double>(clamped_value));
}

/**
 * @brief Convert an OpenMesh point into an Eigen 3D vector for linear algebra.
 */
Eigen::Vector3d point_to_eigen(const MeshType::Point& point)
{
    return Eigen::Vector3d(point[0], point[1], point[2]);
}

/**
 * @brief Extract exactly three vertex ids from a face and reject non-triangle faces.
 */
bool collect_triangle_vertices(
    const MeshType& mesh,
    const MeshType::FaceHandle& face,
    std::array<int, 3>& vertex_ids)
{
    int index = 0;
    for (const auto& vertex_handle : mesh.fv_range(face)) {
        if (index >= 3) {
            return false;
        }
        vertex_ids[index++] = vertex_handle.idx();
    }

    return index == 3;
}

/**
 * @brief Build the 2D local rest-frame edge matrix of a triangle from the 3D mesh.
 */
Eigen::Matrix2d build_rest_edge_matrix(
    const MeshType& mesh,
    const std::array<int, 3>& vertex_ids)
{
    const auto p0 = point_to_eigen(mesh.point(mesh.vertex_handle(vertex_ids[0])));
    const auto p1 = point_to_eigen(mesh.point(mesh.vertex_handle(vertex_ids[1])));
    const auto p2 = point_to_eigen(mesh.point(mesh.vertex_handle(vertex_ids[2])));

    const Eigen::Vector3d edge01 = p1 - p0;
    const Eigen::Vector3d edge02 = p2 - p0;

    const double edge01_norm = edge01.norm();
    if (edge01_norm <= 1e-12) {
        return Eigen::Matrix2d::Zero();
    }

    const Eigen::Vector3d tangent_x = edge01 / edge01_norm;
    Eigen::Vector3d normal = edge01.cross(edge02);
    const double normal_norm = normal.norm();
    if (normal_norm <= 1e-12) {
        return Eigen::Matrix2d::Zero();
    }

    normal /= normal_norm;
    const Eigen::Vector3d tangent_y = normal.cross(tangent_x);

    Eigen::Matrix2d rest_edges;
    rest_edges.col(0) = Eigen::Vector2d(edge01_norm, 0.0);
    rest_edges.col(1) = Eigen::Vector2d(
        edge02.dot(tangent_x),
        edge02.dot(tangent_y));
    return rest_edges;
}

/**
 * @brief Recover triangle vertex coordinates in its local 2D rest frame.
 */
std::array<Eigen::Vector2d, 3> build_rest_triangle_coordinates(
    const Eigen::Matrix2d& rest_edges)
{
    return {
        Eigen::Vector2d::Zero(),
        rest_edges.col(0),
        rest_edges.col(1),
    };
}

/**
 * @brief Compute the cotangent of the angle spanned by two 2D edge vectors.
 */
double compute_cotangent_2d(
    const Eigen::Vector2d& lhs,
    const Eigen::Vector2d& rhs)
{
    const double cross = lhs.x() * rhs.y() - lhs.y() * rhs.x();
    const double cross_abs = std::max(std::abs(cross), 1e-12);
    return lhs.dot(rhs) / cross_abs;
}

/**
 * @brief Project a 2x2 Jacobian onto the closest proper rotation by SVD.
 */
Eigen::Matrix2d compute_closest_rotation(const Eigen::Matrix2d& jacobian)
{
    if (jacobian.squaredNorm() <= 1e-24) {
        return Eigen::Matrix2d::Identity();
    }

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(
        jacobian,
        Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix2d rotation =
        svd.matrixU() * svd.matrixV().transpose();
    if (rotation.determinant() < 0.0) {
        Eigen::Matrix2d adjusted_u = svd.matrixU();
        adjusted_u.col(1) *= -1.0;
        rotation = adjusted_u * svd.matrixV().transpose();
    }

    return rotation;
}

/**
 * @brief Compute the cotangent opposite to the directed halfedge (from, to).
 */
double compute_cotangent(
    const MeshType& mesh,
    MeshType::VertexHandle from,
    MeshType::VertexHandle to)
{
    const auto halfedge = mesh.find_halfedge(from, to);
    if (!halfedge.is_valid() || mesh.is_boundary(halfedge)) {
        return 0.0;
    }

    const auto next_halfedge = mesh.next_halfedge_handle(halfedge);
    const auto opposite_vertex = mesh.to_vertex_handle(next_halfedge);

    const auto from_point = mesh.point(from);
    const auto to_point = mesh.point(to);
    const auto opposite_point = mesh.point(opposite_vertex);

    const auto vec1 = from_point - opposite_point;
    const auto vec2 = to_point - opposite_point;

    double cross_norm = static_cast<double>(vec1.cross(vec2).norm());
    cross_norm = std::max(1e-8, cross_norm);

    return static_cast<double>(vec1.dot(vec2)) / cross_norm;
}

/**
 * @brief Compute the standard cotangent edge weight $\frac{1}{2}(\cot\alpha+\cot\beta)$.
 */
double compute_edge_cotangent_weight(
    const MeshType& mesh,
    MeshType::VertexHandle vh0,
    MeshType::VertexHandle vh1)
{
    return 0.5 *
           (compute_cotangent(mesh, vh0, vh1) +
            compute_cotangent(mesh, vh1, vh0));
}

/**
 * @brief Check whether two meshes share the same connectivity for initialization reuse.
 */
bool meshes_have_compatible_topology(const MeshType& lhs, const MeshType& rhs)
{
    if (lhs.n_vertices() != rhs.n_vertices() ||
        lhs.n_edges() != rhs.n_edges() ||
        lhs.n_faces() != rhs.n_faces()) {
        return false;
    }

    return collect_triangle_signatures(lhs) == collect_triangle_signatures(rhs);
}

/**
 * @brief Use mesh vertex positions as an initial UV buffer.
 */
std::vector<glm::vec2> extract_uv_from_mesh(const MeshType& mesh)
{
    std::vector<glm::vec2> init_uv(mesh.n_vertices());
    for (const auto& vh : mesh.vertices()) {
        const auto point = mesh.point(vh);
        init_uv[vh.idx()] = glm::vec2(point[0], point[1]);
    }
    return init_uv;
}

/**
 * @brief Collect all boundary vertices for fixed-point selection.
 */
std::vector<int> collect_boundary_vertices(const MeshType& mesh)
{
    std::vector<int> boundary_vertices;
    boundary_vertices.reserve(mesh.n_vertices());

    for (const auto& vh : mesh.vertices()) {
        if (mesh.is_boundary(vh)) {
            boundary_vertices.push_back(vh.idx());
        }
    }

    return boundary_vertices;
}

double compute_signed_triangle_area(
    const std::vector<glm::vec2>& uv,
    const std::array<int, 3>& vertex_ids)
{
    const auto& uv0 = uv[vertex_ids[0]];
    const auto& uv1 = uv[vertex_ids[1]];
    const auto& uv2 = uv[vertex_ids[2]];

    const double edge01_x = static_cast<double>(uv1.x - uv0.x);
    const double edge01_y = static_cast<double>(uv1.y - uv0.y);
    const double edge02_x = static_cast<double>(uv2.x - uv0.x);
    const double edge02_y = static_cast<double>(uv2.y - uv0.y);

    return 0.5 * (edge01_x * edge02_y - edge01_y * edge02_x);
}

UvQualityStats evaluate_uv_quality(
    const std::vector<glm::vec2>& uv,
    const std::vector<FacePrecomputation>& face_data)
{
    UvQualityStats stats;
    for (const auto& face : face_data) {
        if (!face.is_valid) {
            continue;
        }

        const double signed_area = compute_signed_triangle_area(uv, face.vertex_ids);
        if (signed_area < -1e-12) {
            ++stats.flipped_face_count;
        }
        else if (std::abs(signed_area) <= 1e-12) {
            ++stats.degenerate_face_count;
        }
    }

    return stats;
}

/**
 * @brief Cache one triangle's rest positions and cotangent weights.
 */
FacePrecomputation build_face_precomputation(
    const MeshType& mesh,
    const MeshType::FaceHandle& face_handle)
{
    FacePrecomputation face_data;
    if (!collect_triangle_vertices(mesh, face_handle, face_data.vertex_ids)) {
        throw std::runtime_error("ARAP: Expected a triangulated mesh.");
    }

    const Eigen::Matrix2d rest_edges =
        build_rest_edge_matrix(mesh, face_data.vertex_ids);
    if (std::abs(rest_edges.determinant()) <= 1e-12) {
        return face_data;
    }

    face_data.rest_positions = build_rest_triangle_coordinates(rest_edges);
    face_data.opposite_edges[0] =
        face_data.rest_positions[2] - face_data.rest_positions[1];
    face_data.opposite_edges[1] =
        face_data.rest_positions[0] - face_data.rest_positions[2];
    face_data.opposite_edges[2] =
        face_data.rest_positions[1] - face_data.rest_positions[0];
    face_data.doubled_area = std::abs(rest_edges.determinant());
    face_data.cotangent_weights[0] = compute_cotangent_2d(
        face_data.rest_positions[1] - face_data.rest_positions[0],
        face_data.rest_positions[2] - face_data.rest_positions[0]);
    face_data.cotangent_weights[1] = compute_cotangent_2d(
        face_data.rest_positions[2] - face_data.rest_positions[1],
        face_data.rest_positions[0] - face_data.rest_positions[1]);
    face_data.cotangent_weights[2] = compute_cotangent_2d(
        face_data.rest_positions[0] - face_data.rest_positions[2],
        face_data.rest_positions[1] - face_data.rest_positions[2]);
    face_data.is_valid = true;
    return face_data;
}

class ArapSolver {
public:
    ArapSolver(
        const MeshType& mesh,
        std::vector<glm::vec2> init_uv,
        ArapOptions options)
        : mesh_(mesh), options_(options)
    {
        state_.current_uv = std::move(init_uv);
        precomputation_.num_vertices = static_cast<int>(mesh_.n_vertices());
        precomputation_.num_faces = static_cast<int>(mesh_.n_faces());
        state_.local_transforms.assign(
            precomputation_.num_faces,
            Eigen::Matrix2d::Identity());
    }

    int flipped_face_count() const;
    int degenerate_face_count() const;

    std::vector<glm::vec2> solve()
    {
        select_fixed_points();
        precompute_face_data();

        if (options_.algorithm == ParameterizationAlgorithm::ASAP) {
            solve_asap();
            update_quality_stats();
            return state_.current_uv;
        }

        precompute_global_system();
        update_quality_stats();

        run_iterations();

        return state_.current_uv;
    }

private:
    const MeshType& mesh_;
    ArapOptions options_;
    ArapConstraints constraints_;
    ArapPrecomputation precomputation_;
    ArapState state_;

    void select_fixed_points();
    void precompute_face_data();
    void precompute_global_system();
    void local_phase();
    bool global_phase();
    void solve_asap();
    void run_iterations();
    void assemble_global_rhs(Eigen::VectorXd& rhs_x, Eigen::VectorXd& rhs_y) const;
    void solve_soft_global_step(const Eigen::VectorXd& rhs_x, const Eigen::VectorXd& rhs_y);
    void solve_hard_global_step(const Eigen::VectorXd& rhs_x, const Eigen::VectorXd& rhs_y);
    void write_full_uv_solution(const Eigen::VectorXd& solved_x, const Eigen::VectorXd& solved_y);
    void write_hard_constraint_solution(
        const Eigen::VectorXd& solved_free_x,
        const Eigen::VectorXd& solved_free_y,
        const Eigen::VectorXd& fixed_x,
        const Eigen::VectorXd& fixed_y);
    void update_quality_stats();
    void precompute_hard_constraint_system();
    void precompute_soft_constraint_system();
};

/**
 * @brief Select two farthest boundary vertices and pin them to their current UV positions.
 */
void ArapSolver::select_fixed_points()
{
    constraints_.fixed_vertex_ids.clear();
    constraints_.fixed_positions.clear();

    const auto boundary_vertices = collect_boundary_vertices(mesh_);
    if (boundary_vertices.empty() || state_.current_uv.empty()) {
        return;
    }

    int first_id = boundary_vertices.front();
    int second_id = first_id;
    double max_distance_sq = -1.0;

    for (size_t i = 0; i < boundary_vertices.size(); ++i) {
        const auto pi = mesh_.point(mesh_.vertex_handle(boundary_vertices[i]));
        for (size_t j = i + 1; j < boundary_vertices.size(); ++j) {
            const auto pj = mesh_.point(mesh_.vertex_handle(boundary_vertices[j]));
            const double dx = static_cast<double>(pi[0]) - static_cast<double>(pj[0]);
            const double dy = static_cast<double>(pi[1]) - static_cast<double>(pj[1]);
            const double dz = static_cast<double>(pi[2]) - static_cast<double>(pj[2]);
            const double distance_sq = dx * dx + dy * dy + dz * dz;

            if (distance_sq > max_distance_sq) {
                max_distance_sq = distance_sq;
                first_id = boundary_vertices[i];
                second_id = boundary_vertices[j];
            }
        }
    }

    constraints_.fixed_vertex_ids.push_back(first_id);
    constraints_.fixed_positions.emplace_back(
        state_.current_uv[first_id].x,
        state_.current_uv[first_id].y);

    if (second_id != first_id) {
        constraints_.fixed_vertex_ids.push_back(second_id);
        constraints_.fixed_positions.emplace_back(
            state_.current_uv[second_id].x,
            state_.current_uv[second_id].y);
    }
}

void ArapSolver::precompute_face_data()
{
    precomputation_.face_data.clear();
    precomputation_.face_data.reserve(static_cast<size_t>(precomputation_.num_faces));

    for (const auto& face_handle : mesh_.faces()) {
        precomputation_.face_data.push_back(
            build_face_precomputation(mesh_, face_handle));
    }
}

void ArapSolver::update_quality_stats()
{
    state_.quality_stats =
        evaluate_uv_quality(state_.current_uv, precomputation_.face_data);
}

int ArapSolver::flipped_face_count() const
{
    return state_.quality_stats.flipped_face_count;
}

int ArapSolver::degenerate_face_count() const
{
    return state_.quality_stats.degenerate_face_count;
}

void ArapSolver::run_iterations()
{
    for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
        local_phase();
        if (!global_phase()) {
            break;
        }
    }
}

void ArapSolver::solve_asap()
{
    const int num_vertices = precomputation_.num_vertices;
    const int num_faces = precomputation_.num_faces;

    if (num_vertices <= 0 || num_faces <= 0) {
        return;
    }

    const ConstraintIndexMaps constraint_maps =
        build_constraint_index_maps(constraints_, num_vertices);
    const int num_fixed = static_cast<int>(
        std::count(
            constraint_maps.is_fixed.begin(),
            constraint_maps.is_fixed.end(),
            static_cast<char>(1)));
    if (num_fixed < 2 ||
        constraints_.fixed_positions.size() < static_cast<size_t>(num_fixed)) {
        throw std::runtime_error("ARAP: ASAP solve requires at least two fixed points.");
    }

    const FreeVertexMapping free_mapping =
        build_free_vertex_mapping(constraint_maps.is_fixed, num_vertices);
    const int free_count = static_cast<int>(free_mapping.free_vertex_ids.size());

    if (free_count == 0) {
        for (int fixed_index = 0; fixed_index < num_fixed; ++fixed_index) {
            const int vertex_id = constraints_.fixed_vertex_ids[fixed_index];
            state_.current_uv[vertex_id] = glm::vec2(
                static_cast<float>(constraints_.fixed_positions[fixed_index].x()),
                static_cast<float>(constraints_.fixed_positions[fixed_index].y()));
        }
        return;
    }

    Eigen::VectorXd fixed_values = Eigen::VectorXd::Zero(2 * num_fixed);
    for (int fixed_index = 0; fixed_index < num_fixed; ++fixed_index) {
        fixed_values[fixed_index] = constraints_.fixed_positions[fixed_index].x();
        fixed_values[fixed_index + num_fixed] = constraints_.fixed_positions[fixed_index].y();
    }

    Eigen::SparseMatrix<double> A(2 * num_faces, 2 * free_count);
    Eigen::SparseMatrix<double> B(2 * num_faces, 2 * num_fixed);
    std::vector<Eigen::Triplet<double>> a_triplets;
    std::vector<Eigen::Triplet<double>> b_triplets;
    a_triplets.reserve(static_cast<size_t>(num_faces) * 12);
    b_triplets.reserve(static_cast<size_t>(num_faces) * 12);

    for (int face_index = 0; face_index < num_faces; ++face_index) {
        const auto& face_data = precomputation_.face_data[face_index];
        if (!face_data.is_valid || face_data.doubled_area <= 1e-12) {
            continue;
        }

        const double coeff = std::sqrt(face_data.doubled_area);
        for (int local_index = 0; local_index < 3; ++local_index) {
            const int vertex_id = face_data.vertex_ids[local_index];
            const double dx = face_data.opposite_edges[local_index].x() / coeff;
            const double dy = face_data.opposite_edges[local_index].y() / coeff;

            const int fixed_index = constraint_maps.fixed_slot[vertex_id];
            if (fixed_index >= 0) {
                b_triplets.emplace_back(face_index, fixed_index, dx);
                b_triplets.emplace_back(face_index + num_faces, fixed_index, dy);
                b_triplets.emplace_back(face_index, fixed_index + num_fixed, -dy);
                b_triplets.emplace_back(face_index + num_faces, fixed_index + num_fixed, dx);
                continue;
            }

            const int free_index = free_mapping.full_to_free[vertex_id];
            if (free_index < 0) {
                continue;
            }

            a_triplets.emplace_back(face_index, free_index, dx);
            a_triplets.emplace_back(face_index + num_faces, free_index, dy);
            a_triplets.emplace_back(face_index, free_index + free_count, -dy);
            a_triplets.emplace_back(face_index + num_faces, free_index + free_count, dx);
        }
    }

    A.setFromTriplets(a_triplets.begin(), a_triplets.end());
    B.setFromTriplets(b_triplets.begin(), b_triplets.end());
    A.makeCompressed();
    B.makeCompressed();

    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("ARAP: Failed to factorize ASAP least-squares system.");
    }

    const Eigen::VectorXd rhs = -B * fixed_values;
    const Eigen::VectorXd solution = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("ARAP: Failed to solve ASAP least-squares system.");
    }

    for (int vertex_id = 0; vertex_id < num_vertices; ++vertex_id) {
        const int fixed_index = constraint_maps.fixed_slot[vertex_id];
        if (fixed_index >= 0) {
            state_.current_uv[vertex_id] = glm::vec2(
                static_cast<float>(constraints_.fixed_positions[fixed_index].x()),
                static_cast<float>(constraints_.fixed_positions[fixed_index].y()));
            continue;
        }

        const int free_index = free_mapping.full_to_free[vertex_id];
        state_.current_uv[vertex_id] = glm::vec2(
            static_cast<float>(solution[free_index]),
            static_cast<float>(solution[free_index + free_count]));
    }
}

/**
 * @brief Assemble and factorize the global system for the selected constraint mode.
 */
void ArapSolver::precompute_global_system()
{
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(precomputation_.num_vertices) * 8);

    for (const auto& vh : mesh_.vertices()) {
        double diagonal_sum = 0.0;

        for (const auto& vv_it : mesh_.vv_range(vh)) {
            const double weight = compute_edge_cotangent_weight(mesh_, vh, vv_it);
            if (std::abs(weight) <= 1e-12) {
                continue;
            }

            diagonal_sum += weight;
            triplets.emplace_back(vh.idx(), vv_it.idx(), -weight);
        }

        triplets.emplace_back(vh.idx(), vh.idx(), diagonal_sum);
    }

    precomputation_.laplacian_matrix.resize(
        precomputation_.num_vertices,
        precomputation_.num_vertices);
    precomputation_.laplacian_matrix.setFromTriplets(
        triplets.begin(),
        triplets.end());
    precomputation_.laplacian_matrix.makeCompressed();

    if (options_.constraint_mode == ConstraintMode::HardElimination) {
        precompute_hard_constraint_system();
    }
    else {
        precompute_soft_constraint_system();
    }
}

void ArapSolver::precompute_soft_constraint_system()
{
    precomputation_.system_matrix = precomputation_.laplacian_matrix;
    for (const int fixed_vertex_id : constraints_.fixed_vertex_ids) {
        if (fixed_vertex_id < 0 || fixed_vertex_id >= precomputation_.num_vertices) {
            continue;
        }

        precomputation_.system_matrix.coeffRef(
            fixed_vertex_id,
            fixed_vertex_id) += options_.soft_penalty_weight;
    }
    precomputation_.system_matrix.makeCompressed();

    precomputation_.solver.compute(precomputation_.system_matrix);
    if (precomputation_.solver.info() != Eigen::Success) {
        throw std::runtime_error("ARAP: Failed to factorize soft-constraint system.");
    }
}

void ArapSolver::precompute_hard_constraint_system()
{
    const ConstraintIndexMaps constraint_maps =
        build_constraint_index_maps(constraints_, precomputation_.num_vertices);
    const FreeVertexMapping free_mapping =
        build_free_vertex_mapping(
            constraint_maps.is_fixed,
            precomputation_.num_vertices);
    precomputation_.free_vertex_ids = free_mapping.free_vertex_ids;
    precomputation_.full_to_free = free_mapping.full_to_free;

    std::vector<Eigen::Triplet<double>> reduced_triplets;
    reduced_triplets.reserve(
        static_cast<size_t>(precomputation_.laplacian_matrix.nonZeros()));

    for (int outer_index = 0;
         outer_index < precomputation_.laplacian_matrix.outerSize();
         ++outer_index) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(
                 precomputation_.laplacian_matrix,
                 outer_index);
             it;
             ++it) {
            const int free_row = precomputation_.full_to_free[it.row()];
            const int free_col = precomputation_.full_to_free[it.col()];
            if (free_row < 0 || free_col < 0) {
                continue;
            }

            reduced_triplets.emplace_back(free_row, free_col, it.value());
        }
    }

    precomputation_.system_matrix.resize(
        static_cast<int>(precomputation_.free_vertex_ids.size()),
        static_cast<int>(precomputation_.free_vertex_ids.size()));
    precomputation_.system_matrix.setFromTriplets(
        reduced_triplets.begin(),
        reduced_triplets.end());
    precomputation_.system_matrix.makeCompressed();

    precomputation_.solver.compute(precomputation_.system_matrix);
    if (precomputation_.solver.info() != Eigen::Success) {
        throw std::runtime_error("ARAP: Failed to factorize hard-constraint system.");
    }
}

/**
 * @brief Compute one closest rotation matrix for each triangle from the current UV Jacobian.
 */
void ArapSolver::local_phase()
{
    if (state_.current_uv.size() != static_cast<size_t>(precomputation_.num_vertices)) {
        throw std::runtime_error("ARAP: UV state size does not match vertex count.");
    }

    for (size_t face_index = 0;
         face_index < precomputation_.face_data.size();
         ++face_index) {
        const auto& face_data = precomputation_.face_data[face_index];
        if (!face_data.is_valid) {
            state_.local_transforms[face_index] = Eigen::Matrix2d::Identity();
            continue;
        }

        if (options_.algorithm == ParameterizationAlgorithm::Hybrid) {
            state_.local_transforms[face_index] =
                compute_hybrid_local_transform(
                    face_data,
                    state_.current_uv,
                    options_.hybrid_lambda);
            continue;
        }

        Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();

        auto accumulate_covariance = [&](int local_from, int local_to, double cotangent) {
            if (std::abs(cotangent) <= 1e-12) {
                return;
            }

            const int vertex_from = face_data.vertex_ids[local_from];
            const int vertex_to = face_data.vertex_ids[local_to];
            const auto& uv_from = state_.current_uv[vertex_from];
            const auto& uv_to = state_.current_uv[vertex_to];

            const Eigen::Vector2d uv_edge(
                uv_from.x - uv_to.x,
                uv_from.y - uv_to.y);
            const Eigen::Vector2d rest_edge =
                face_data.rest_positions[local_from] -
                face_data.rest_positions[local_to];

            covariance +=
                0.5 * cotangent * uv_edge * rest_edge.transpose();
        };

        accumulate_covariance(0, 1, face_data.cotangent_weights[2]);
        accumulate_covariance(1, 2, face_data.cotangent_weights[0]);
        accumulate_covariance(2, 0, face_data.cotangent_weights[1]);

        state_.local_transforms[face_index] =
            compute_closest_rotation(covariance);
    }
}

/**
 * @brief Assemble the global right-hand side with fixed local rotations and solve updated UVs.
 */
bool ArapSolver::global_phase()
{
    if (state_.local_transforms.size() != static_cast<size_t>(precomputation_.num_faces)) {
        throw std::runtime_error("ARAP: Local transform state size does not match face count.");
    }

    Eigen::VectorXd rhs_x = Eigen::VectorXd::Zero(precomputation_.num_vertices);
    Eigen::VectorXd rhs_y = Eigen::VectorXd::Zero(precomputation_.num_vertices);

    assemble_global_rhs(rhs_x, rhs_y);

    if (options_.constraint_mode == ConstraintMode::SoftPenalty) {
        solve_soft_global_step(rhs_x, rhs_y);
    }
    else {
        solve_hard_global_step(rhs_x, rhs_y);
    }

    update_quality_stats();

    return true;
}

void ArapSolver::assemble_global_rhs(Eigen::VectorXd& rhs_x, Eigen::VectorXd& rhs_y) const
{
    if (rhs_x.size() != precomputation_.num_vertices ||
        rhs_y.size() != precomputation_.num_vertices) {
        throw std::runtime_error("ARAP: RHS buffers do not match vertex count.");
    }

    auto accumulate_edge_contribution =
        [&](int from, int to, double cotangent, const Eigen::Matrix2d& local_transform, const std::array<Eigen::Vector2d, 3>& rest_positions, int local_from, int local_to) {
            if (std::abs(cotangent) <= 1e-12) {
                return;
            }

            const Eigen::Vector2d rotated_edge =
                0.5 * cotangent * local_transform * (rest_positions[local_from] - rest_positions[local_to]);

            rhs_x[from] += rotated_edge.x();
            rhs_y[from] += rotated_edge.y();
            rhs_x[to] -= rotated_edge.x();
            rhs_y[to] -= rotated_edge.y();
        };

    for (size_t face_index = 0;
         face_index < precomputation_.face_data.size();
         ++face_index) {
        const auto& face_data = precomputation_.face_data[face_index];
        if (!face_data.is_valid) {
            continue;
        }

        const Eigen::Matrix2d& local_transform = state_.local_transforms[face_index];
        accumulate_edge_contribution(
            face_data.vertex_ids[0],
            face_data.vertex_ids[1],
            face_data.cotangent_weights[2],
            local_transform,
            face_data.rest_positions,
            0,
            1);
        accumulate_edge_contribution(
            face_data.vertex_ids[1],
            face_data.vertex_ids[2],
            face_data.cotangent_weights[0],
            local_transform,
            face_data.rest_positions,
            1,
            2);
        accumulate_edge_contribution(
            face_data.vertex_ids[2],
            face_data.vertex_ids[0],
            face_data.cotangent_weights[1],
            local_transform,
            face_data.rest_positions,
            2,
            0);
    }
}

void ArapSolver::solve_soft_global_step(
    const Eigen::VectorXd& rhs_x,
    const Eigen::VectorXd& rhs_y)
{
    Eigen::VectorXd constrained_rhs_x = rhs_x;
    Eigen::VectorXd constrained_rhs_y = rhs_y;

    for (size_t constraint_index = 0;
         constraint_index < constraints_.fixed_vertex_ids.size() &&
         constraint_index < constraints_.fixed_positions.size();
         ++constraint_index) {
        const int vertex_id = constraints_.fixed_vertex_ids[constraint_index];
        if (vertex_id < 0 || vertex_id >= precomputation_.num_vertices) {
            continue;
        }

        constrained_rhs_x[vertex_id] +=
            options_.soft_penalty_weight * constraints_.fixed_positions[constraint_index].x();
        constrained_rhs_y[vertex_id] +=
            options_.soft_penalty_weight * constraints_.fixed_positions[constraint_index].y();
    }

    const Eigen::VectorXd solved_x = precomputation_.solver.solve(constrained_rhs_x);
    if (precomputation_.solver.info() != Eigen::Success) {
        throw std::runtime_error("ARAP: Failed to solve x-component in soft global phase.");
    }

    const Eigen::VectorXd solved_y = precomputation_.solver.solve(constrained_rhs_y);
    if (precomputation_.solver.info() != Eigen::Success) {
        throw std::runtime_error("ARAP: Failed to solve y-component in soft global phase.");
    }

    write_full_uv_solution(solved_x, solved_y);
}

void ArapSolver::solve_hard_global_step(
    const Eigen::VectorXd& rhs_x,
    const Eigen::VectorXd& rhs_y)
{
    Eigen::VectorXd fixed_x = Eigen::VectorXd::Zero(precomputation_.num_vertices);
    Eigen::VectorXd fixed_y = Eigen::VectorXd::Zero(precomputation_.num_vertices);
    for (size_t constraint_index = 0;
         constraint_index < constraints_.fixed_vertex_ids.size() &&
         constraint_index < constraints_.fixed_positions.size();
         ++constraint_index) {
        const int vertex_id = constraints_.fixed_vertex_ids[constraint_index];
        if (vertex_id < 0 || vertex_id >= precomputation_.num_vertices) {
            continue;
        }

        fixed_x[vertex_id] = constraints_.fixed_positions[constraint_index].x();
        fixed_y[vertex_id] = constraints_.fixed_positions[constraint_index].y();
    }

    const Eigen::VectorXd adjusted_rhs_x =
        rhs_x - precomputation_.laplacian_matrix * fixed_x;
    const Eigen::VectorXd adjusted_rhs_y =
        rhs_y - precomputation_.laplacian_matrix * fixed_y;

    Eigen::VectorXd reduced_rhs_x(
        static_cast<int>(precomputation_.free_vertex_ids.size()));
    Eigen::VectorXd reduced_rhs_y(
        static_cast<int>(precomputation_.free_vertex_ids.size()));
    for (size_t free_index = 0;
         free_index < precomputation_.free_vertex_ids.size();
         ++free_index) {
        const int full_index = precomputation_.free_vertex_ids[free_index];
        reduced_rhs_x[static_cast<int>(free_index)] = adjusted_rhs_x[full_index];
        reduced_rhs_y[static_cast<int>(free_index)] = adjusted_rhs_y[full_index];
    }

    const Eigen::VectorXd solved_free_x =
        precomputation_.solver.solve(reduced_rhs_x);
    if (precomputation_.solver.info() != Eigen::Success) {
        throw std::runtime_error("ARAP: Failed to solve x-component in hard global phase.");
    }

    const Eigen::VectorXd solved_free_y =
        precomputation_.solver.solve(reduced_rhs_y);
    if (precomputation_.solver.info() != Eigen::Success) {
        throw std::runtime_error("ARAP: Failed to solve y-component in hard global phase.");
    }

    write_hard_constraint_solution(solved_free_x, solved_free_y, fixed_x, fixed_y);
}

void ArapSolver::write_full_uv_solution(
    const Eigen::VectorXd& solved_x,
    const Eigen::VectorXd& solved_y)
{
    for (int vertex_index = 0; vertex_index < precomputation_.num_vertices; ++vertex_index) {
        state_.current_uv[vertex_index] = glm::vec2(
            static_cast<float>(solved_x[vertex_index]),
            static_cast<float>(solved_y[vertex_index]));
    }
}

void ArapSolver::write_hard_constraint_solution(
    const Eigen::VectorXd& solved_free_x,
    const Eigen::VectorXd& solved_free_y,
    const Eigen::VectorXd& fixed_x,
    const Eigen::VectorXd& fixed_y)
{
    for (int vertex_index = 0; vertex_index < precomputation_.num_vertices; ++vertex_index) {
        state_.current_uv[vertex_index] = glm::vec2(0.0f);
    }

    for (size_t constraint_index = 0;
         constraint_index < constraints_.fixed_vertex_ids.size() &&
         constraint_index < constraints_.fixed_positions.size();
         ++constraint_index) {
        const int vertex_id = constraints_.fixed_vertex_ids[constraint_index];
        if (vertex_id < 0 || vertex_id >= precomputation_.num_vertices) {
            continue;
        }

        state_.current_uv[vertex_id] = glm::vec2(
            static_cast<float>(fixed_x[vertex_id]),
            static_cast<float>(fixed_y[vertex_id]));
    }

    for (size_t free_index = 0;
         free_index < precomputation_.free_vertex_ids.size();
         ++free_index) {
        const int full_index = precomputation_.free_vertex_ids[free_index];
        state_.current_uv[full_index] = glm::vec2(
            static_cast<float>(solved_free_x[static_cast<int>(free_index)]),
            static_cast<float>(solved_free_y[static_cast<int>(free_index)]));
    }
}
}  // namespace

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(hw6_arap)
{
    // Original 3D mesh to parameterize. This should be a triangulated mesh
    // with boundary. If the node returns false here, first check whether a
    // MeshComponent exists on the incoming Geometry.
    b.add_input<Geometry>("Input");

    // Initialization mesh. Its vertex x/y coordinates are read as the initial
    // UV guess, so it must have the same topology as Input. In practice this
    // is usually the output of a HW5 parameterization branch.
    b.add_input<Geometry>("InitUV_Mesh");

    // Number of local/global iterations. Only used by ARAP and Hybrid.
    b.add_input<int>("Iterations").default_val(10).min(1).max(100);

    // Parameterization model selector:
    // 0 = ARAP, 1 = ASAP, 2 = Hybrid.
    // Use the same scene and init UV to compare distortion across models.
    b.add_input<int>("Algorithm").default_val(0).min(0).max(2);

    // Hybrid-only parameter lambda in [0, +inf). It is ignored by ARAP and ASAP.
    b.add_input<float>("Hybrid Lambda").default_val(1.0f).min(0.0f).max(100.0f);

    // Constraint handling mode for ARAP / Hybrid global solve.
    // ASAP uses its own least-squares solve and ignores this setting.
    b.add_input<int>("Constraint Mode").default_val(0).min(0).max(1);

    // Log10 of the soft penalty weight. Only used when Algorithm != ASAP and Constraint Mode = 0.
    // Example: 8 means a penalty weight of 10^8.
    b.add_input<int>("Soft Constraint Log10").default_val(8).min(2).max(12);

    // UV buffer to be written back into a mesh via
    // mesh_add_vertex_parameterization_quantity for visualization/debugging.
    b.add_output<std::vector<glm::vec2>>("OutputUV");

    // Count of triangles whose signed UV area becomes negative. A non-zero
    // value usually means flips caused by bad initialization or unstable setup.
    b.add_output<int>("FlippedFaceCount");

    // Count of triangles whose UV area nearly collapses to zero. This is useful
    // for spotting shrinkage or near-singular local/global updates.
    b.add_output<int>("DegenerateFaceCount");
}

NODE_EXECUTION_FUNCTION(hw6_arap)
{
    try {
        // Read node inputs from the execution context. Keep these names aligned
        // with the declaration above, otherwise the GUI sockets and code will
        // silently disagree.
        auto input = params.get_input<Geometry>("Input");
        auto init_uv_mesh = params.get_input<Geometry>("InitUV_Mesh");
        const auto raw_iterations = params.get_input<int>("Iterations");
        const auto raw_algorithm = params.get_input<int>("Algorithm");
        const auto raw_hybrid_lambda = params.get_input<float>("Hybrid Lambda");
        const auto raw_constraint_mode = params.get_input<int>("Constraint Mode");
        const auto raw_soft_constraint_log10 =
            params.get_input<int>("Soft Constraint Log10");
        const auto algorithm = parse_parameterization_algorithm(raw_algorithm);
        const auto iterations = raw_iterations > 0
                        ? std::clamp(raw_iterations, 1, 100)
                        : 10;

        if (!input.get_component<MeshComponent>()) {
            return false;
        }
        if (!init_uv_mesh.get_component<MeshComponent>()) {
            return false;
        }

        auto mesh_3d = operand_to_openmesh(&input);
        auto mesh_uv = operand_to_openmesh(&init_uv_mesh);
        if (!meshes_have_compatible_topology(*mesh_3d, *mesh_uv)) {
            return false;
        }

        ArapOptions options;
        options.algorithm = algorithm;
        options.max_iterations =
            algorithm == ParameterizationAlgorithm::ASAP ? 1 : iterations;
        options.hybrid_lambda =
            algorithm == ParameterizationAlgorithm::Hybrid
                ? std::max(0.0, static_cast<double>(raw_hybrid_lambda))
                : 0.0;
        options.constraint_mode =
            algorithm == ParameterizationAlgorithm::ASAP
                ? ConstraintMode::HardElimination
                : parse_constraint_mode(raw_constraint_mode);
        options.soft_penalty_weight =
            algorithm == ParameterizationAlgorithm::ASAP
                ? 0.0
                : penalty_weight_from_slider(
                      raw_soft_constraint_log10 > 0 ? raw_soft_constraint_log10 : 8);

        auto init_uv = extract_uv_from_mesh(*mesh_uv);
        ArapSolver solver(*mesh_3d, std::move(init_uv), options);
        auto uv_result = solver.solve();

        params.set_output("OutputUV", std::move(uv_result));
        params.set_output("FlippedFaceCount", solver.flipped_face_count());
        params.set_output("DegenerateFaceCount", solver.degenerate_face_count());
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

NODE_DECLARATION_UI(hw6_arap);
NODE_DEF_CLOSE_SCOPE