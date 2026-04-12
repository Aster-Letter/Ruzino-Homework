#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <functional>

#include <stdexcept>
#include <vector>

#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"

/*
** @brief HW4_TutteParameterization
**
** This file presents the basic framework of a "node", which processes inputs
** received from the left and outputs specific variables for downstream nodes to
** use.
** - In the first function, node_declare, you can set up the node's input and
** output variables.
** - The second function, node_exec is the execution part of the node, where we
** need to implement the node's functionality.
** Your task is to fill in the required logic at the specified locations
** within this template, especially in node_exec.
*/

namespace {

using MeshType = OpenMesh::PolyMesh_ArrayKernelT<>;
using VertexHandle = OpenMesh::VertexHandle;

constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-12;

enum class WeightType {
    Uniform,
    Cotangent,
    Floater
};

using WeightList = std::vector<double>;

using WeightFunction = std::function<WeightList(
    const MeshType&,
    const VertexHandle&,
    const std::vector<VertexHandle>&)>;

std::vector<VertexHandle> collect_one_ring_neighbors(
    const MeshType& mesh,
    const VertexHandle& vh)
{
    std::vector<VertexHandle> neighbors;
    for (const auto& neighbor : mesh.vv_range(vh)) {
        neighbors.push_back(neighbor);
    }
    return neighbors;
}

WeightList build_uniform_weight_list(size_t neighbor_count)
{
    return WeightList(neighbor_count, 1.0);
}

WeightList normalize_weight_list(WeightList weights)
{
    const auto weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weight_sum <= 0.0) {
        return weights;
    }

    for (auto& weight : weights) {
        weight /= weight_sum;
    }
    return weights;
}

std::vector<OpenMesh::Vec3f> collect_neighbor_offsets(
    const MeshType& mesh,
    const VertexHandle& center,
    const std::vector<VertexHandle>& neighbors)
{
    std::vector<OpenMesh::Vec3f> offsets;
    offsets.reserve(neighbors.size());

    const auto center_point = mesh.point(center);
    for (const auto& neighbor : neighbors) {
        offsets.push_back(mesh.point(neighbor) - center_point);
    }

    return offsets;
}

double safe_angle_between(const OpenMesh::Vec3f& lhs, const OpenMesh::Vec3f& rhs)
{
    const auto lhs_norm = OpenMesh::norm(lhs);
    const auto rhs_norm = OpenMesh::norm(rhs);
    const auto denominator = std::max(
        kEpsilon,
        static_cast<double>(lhs_norm) * static_cast<double>(rhs_norm));
    const auto cosine = std::clamp(
        static_cast<double>(OpenMesh::dot(lhs, rhs)) / denominator,
        -1.0,
        1.0);
    return std::acos(cosine);
}

double cross2d(const Eigen::Vector2d& lhs, const Eigen::Vector2d& rhs)
{
    return lhs.x() * rhs.y() - lhs.y() * rhs.x();
}

bool build_floater_planar_ring(
    const std::vector<OpenMesh::Vec3f>& offset_vectors,
    const std::vector<double>& turning_angles,
    std::vector<Eigen::Vector2d>& planar_ring)
{
    const auto degree = offset_vectors.size();
    if (degree < 3 || degree != turning_angles.size()) {
        return false;
    }

    const auto sum_angle =
        std::accumulate(turning_angles.begin(), turning_angles.end(), 0.0);
    if (sum_angle <= kEpsilon) {
        return false;
    }

    planar_ring.resize(degree);
    auto current_theta = 0.0;
    for (size_t i = 0; i < degree; ++i) {
        const auto length = static_cast<double>(offset_vectors[i].norm());
        if (length <= kEpsilon) {
            return false;
        }

        planar_ring[i] = Eigen::Vector2d(
            length * std::cos(current_theta),
            length * std::sin(current_theta));
        current_theta += turning_angles[i] * (2.0 * kPi / sum_angle);
    }

    return true;
}

bool find_floater_visible_edge(
    const std::vector<Eigen::Vector2d>& planar_ring,
    size_t source_index,
    size_t& left_index,
    size_t& right_index)
{
    const auto degree = planar_ring.size();
    const auto ray_direction = -planar_ring[source_index];

    left_index = (source_index + 1) % degree;
    right_index = (left_index + 1) % degree;

    for (size_t step = 0; step < degree; ++step) {
        const auto left_cross = cross2d(planar_ring[left_index], ray_direction);
        const auto right_cross = cross2d(planar_ring[right_index], ray_direction);
        if (left_cross >= -kEpsilon && right_cross <= kEpsilon) {
            return true;
        }

        left_index = right_index;
        right_index = (right_index + 1) % degree;
        if (left_index == source_index) {
            break;
        }
    }

    return false;
}

bool compute_floater_local_barycentrics(
    const std::vector<Eigen::Vector2d>& planar_ring,
    size_t source_index,
    size_t left_index,
    size_t right_index,
    double& source_weight,
    double& left_weight,
    double& right_weight)
{
    const auto& source = planar_ring[source_index];
    const auto ray_direction = -source;
    const auto& left = planar_ring[left_index];
    const auto& right = planar_ring[right_index];
    const auto edge = right - left;

    const auto denominator = cross2d(ray_direction, edge);
    if (std::abs(denominator) <= kEpsilon) {
        return false;
    }

    const auto ray_scale = cross2d(left, edge) / denominator;
    const auto segment_t = cross2d(left, ray_direction) / denominator;
    if (ray_scale <= kEpsilon || segment_t < -kEpsilon || segment_t > 1.0 + kEpsilon) {
        return false;
    }

    const auto clamped_t = std::clamp(segment_t, 0.0, 1.0);
    const auto normalization = 1.0 + ray_scale;

    source_weight = ray_scale / normalization;
    left_weight = (1.0 - clamped_t) / normalization;
    right_weight = clamped_t / normalization;
    return true;
}

std::vector<double> compute_cyclic_one_ring_angles(
    const MeshType& mesh,
    const VertexHandle& center,
    const std::vector<VertexHandle>& neighbors)
{
    std::vector<double> angles(neighbors.size(), 0.0);
    if (neighbors.size() < 2) {
        return angles;
    }

    const auto offsets = collect_neighbor_offsets(mesh, center, neighbors);
    for (size_t i = 0; i < offsets.size(); ++i) {
        angles[i] = safe_angle_between(offsets[i], offsets[(i + 1) % offsets.size()]);
    }

    return angles;
}

bool meshes_have_compatible_topology(const MeshType& lhs, const MeshType& rhs)
{
    return lhs.n_vertices() == rhs.n_vertices() &&
           lhs.n_edges() == rhs.n_edges() &&
           lhs.n_faces() == rhs.n_faces();
}

WeightType parse_weight_type(int raw_weight_type)
{
    switch (std::clamp(raw_weight_type, 0, 2)) {
        case 0:
            return WeightType::Uniform;
        case 1:
            return WeightType::Cotangent;
        case 2:
            return WeightType::Floater;
        default:
            throw std::runtime_error("Invalid weight type.");
    }
}

// keep the form of compute_edge_weight.
WeightList get_edge_uniform_weight(
    const MeshType& mesh,
    const VertexHandle& vh,
    const std::vector<VertexHandle>& neighbors)
{
    (void)mesh;
    (void)vh;
    return build_uniform_weight_list(neighbors.size());
}

// Further encapsulation for cotangent weight to calculate the cotangent of the angle opposite to the specific half_edge (vh1, vh2).
double compute_cotangent(
    const MeshType& mesh,
    const VertexHandle& vh1,
    const VertexHandle& vh2)
{
    auto heh = mesh.find_halfedge(vh1, vh2);
    if (heh.is_valid() && !mesh.is_boundary(heh)) {

        auto heh_next = mesh.next_halfedge_handle(heh);
        auto vh_opposite = mesh.to_vertex_handle(heh_next); 

        auto v1 = mesh.point(vh1);
        auto v2 = mesh.point(vh2);
        auto v_opposite = mesh.point(vh_opposite);


        auto vec1 = v1 - v_opposite;
        auto vec2 = v2 - v_opposite;
        
        // To avoid numerical instability when the angle is close to 0 or 180 degrees, we can add a small epsilon to the denominator.
        double cross_norm = static_cast<double>(vec1.cross(vec2).norm());
        cross_norm = std::max(1e-8, cross_norm); 
        
        return vec1.dot(vec2) / cross_norm;
    }
    return 0.0;
}

double compute_edge_cotangent_weight(
    const MeshType& mesh,
    const VertexHandle& vh1,
    const VertexHandle& vh2)
{
    double weight = 0.0;

    auto heh1 = mesh.find_halfedge(vh1, vh2);
    if (heh1.is_valid() && !mesh.is_boundary(heh1)) {
        weight += compute_cotangent(mesh, vh1, vh2);
    }

    auto heh2 = mesh.find_halfedge(vh2, vh1);
    if (heh2.is_valid() && !mesh.is_boundary(heh2)) {
        weight += compute_cotangent(mesh, vh2, vh1);
    }

    weight /= 2.0;
    return weight;
}

WeightList get_edge_cotangent_weight(
    const MeshType& mesh,
    const VertexHandle& vh,
    const std::vector<VertexHandle>& neighbors)
{
    WeightList weights;
    weights.reserve(neighbors.size());
    for (const auto& neighbor : neighbors) {
        weights.push_back(compute_edge_cotangent_weight(mesh, vh, neighbor));
    }
    return weights;
}

WeightList get_edge_floater_weight(
    const MeshType& mesh,
    const VertexHandle& vh,
    const std::vector<VertexHandle>& neighbors)
{
    const auto turning_angles = compute_cyclic_one_ring_angles(mesh, vh, neighbors);
    const auto offset_vectors = collect_neighbor_offsets(mesh, vh, neighbors);
    const size_t degree = neighbors.size();

    if (degree < 3) {
        return build_uniform_weight_list(degree);
    }

    // Final weights to be computed
    WeightList weights(degree, 0.0);

    std::vector<Eigen::Vector2d> p(degree);
    if (!build_floater_planar_ring(offset_vectors, turning_angles, p)) {
        return build_uniform_weight_list(degree);
    }

    for (size_t i = 0; i < degree; ++i) {
        size_t left_index = 0;
        size_t right_index = 0;
        if (!find_floater_visible_edge(p, i, left_index, right_index)) {
            return build_uniform_weight_list(degree);
        }

        double source_weight = 0.0;
        double left_weight = 0.0;
        double right_weight = 0.0;
        if (!compute_floater_local_barycentrics(
                p,
                i,
                left_index,
                right_index,
                source_weight,
                left_weight,
                right_weight)) {
            return build_uniform_weight_list(degree);
        }

        weights[i] += source_weight;
        if (left_index != i) {
            weights[left_index] += left_weight;
        }
        if (right_index != i) {
            weights[right_index] += right_weight;
        }
    }

    const auto weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weight_sum <= kEpsilon) {
        return build_uniform_weight_list(degree);
    }

    for (const auto weight : weights) {
        if (weight < -kEpsilon) {
            return build_uniform_weight_list(degree);
        }
    }

    return weights;
}

WeightFunction make_weight_strategy(WeightType weight_type, bool normalize_weights)
{
    WeightFunction weight_strategy;
    switch (weight_type) {
        case WeightType::Uniform:
            weight_strategy = get_edge_uniform_weight;
            break;
        case WeightType::Cotangent:
            weight_strategy = get_edge_cotangent_weight;
            break;
        case WeightType::Floater:
            weight_strategy = get_edge_floater_weight;
            break;
    }

    if (!normalize_weights) {
        return weight_strategy;
    }

    return [base_strategy = std::move(weight_strategy)](
               const MeshType& mesh,
               const VertexHandle& vh,
               const std::vector<VertexHandle>& neighbors) {
        return normalize_weight_list(base_strategy(mesh, vh, neighbors));
    };
}


void build_linear_system(
    const MeshType& mesh,
    const MeshType& weight_mesh,
    Eigen::SparseMatrix<double>& A,
    Eigen::VectorXd& b_x,
    Eigen::VectorXd& b_y,
    Eigen::VectorXd& b_z,
    WeightFunction weight_func)
{
    const auto num_vertices = static_cast<int>(mesh.n_vertices());
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(num_vertices * 7);
    auto has_boundary = false;

    for (const auto& vh : mesh.vertices()) {
        if (mesh.is_boundary(vh)) {
            has_boundary = true;
            triplets.emplace_back(vh.idx(), vh.idx(), 1.0);

            const auto point = mesh.point(vh);
            b_x(vh.idx()) = point[0];
            b_y(vh.idx()) = point[1];
            b_z(vh.idx()) = point[2];
            continue;
        }

        double diagonal_weight_sum = 0.0;

        const auto neighbors = collect_one_ring_neighbors(mesh, vh);
        const auto weights = weight_func(weight_mesh, vh, neighbors);

        if (weights.size() != neighbors.size()) {
            throw std::runtime_error(
                "Minimal Surface: Weight count does not match neighbor count.");
        }

        for (size_t neighbor_index = 0; neighbor_index < neighbors.size(); ++neighbor_index) {
            const auto weight_entry = weights[neighbor_index];
            diagonal_weight_sum += weight_entry;
            triplets.emplace_back(
                vh.idx(),
                neighbors[neighbor_index].idx(),
                -weight_entry);
        }

        if (diagonal_weight_sum == 0.0) {
            throw std::runtime_error(
                "Minimal Surface: Interior vertex has no neighbors.");
        }

        triplets.emplace_back(vh.idx(), vh.idx(), diagonal_weight_sum);
    }

    if (!has_boundary) {
        throw std::runtime_error(
            "Minimal Surface: Mesh must contain at least one boundary loop.");
    }

    A.resize(num_vertices, num_vertices);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
}

Eigen::VectorXd solve_linear_system(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b)
{
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Minimal Surface: Matrix decomposition failed.");
    }

    auto x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Minimal Surface: Linear solve failed.");
    }

    return x;
}

void update_vertex_positions(
    OpenMesh::PolyMesh_ArrayKernelT<>& mesh,
    const Eigen::VectorXd& xx,
    const Eigen::VectorXd& yy,
    const Eigen::VectorXd& zz)
{
    for (const auto& vh : mesh.vertices()) {
        auto point = mesh.point(vh);
        point[0] = xx(vh.idx());
        point[1] = yy(vh.idx());
        point[2] = zz(vh.idx());
        mesh.set_point(vh, point);
    }
}

} // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw5_param)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("Reference Mesh");
    b.add_input<int>("Weight Type").default_val(0).min(0).max(2);
    b.add_input<int>("Normalize Weights").default_val(1).min(0).max(1);

    /*
    ** NOTE: You can add more inputs or outputs if necessary. For example, in
    *some cases,
    ** additional information (e.g. other mesh geometry, other parameters) is
    *required to perform
    ** the computation.
    **
    ** Be sure that the input/outputs do not share the same name. You can add
    *one geometry as
    **
    **                b.add_input<Geometry>("Input");
    **
    ** Or maybe you need a value buffer like:
    **
    **                b.add_input<float1Buffer>("Weights");
    */

    // Output-1: Minimal surface with fixed boundary
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_param)
{
    try {
        // Get the input from params
        auto input = params.get_input<Geometry>("Input");
        auto reference_input = params.get_input<Geometry>("Reference Mesh");
        const auto weight_type =
            parse_weight_type(params.get_input<int>("Weight Type"));
        const auto normalize_weights =
            params.get_input<int>("Normalize Weights") != 0;

        // (TO BE UPDATED) Avoid processing the node when there is no input
        if (!input.get_component<MeshComponent>()) {
            return false;
        }

        /* ----------------------------- Preprocess -------------------------------
        ** Create a halfedge structure (using OpenMesh) for the input mesh. The
        ** half-edge data structure is a widely used data structure in geometric
        ** processing, offering convenient operations for traversing and modifying
        ** mesh elements.
        */
        auto halfedge_mesh = operand_to_openmesh(&input);
        std::shared_ptr<MeshType> reference_halfedge_mesh;
        const MeshType* weight_mesh = halfedge_mesh.get();

        if (reference_input.get_component<MeshComponent>()) {
            reference_halfedge_mesh = operand_to_openmesh(&reference_input);
            if (!meshes_have_compatible_topology(
                    *halfedge_mesh, *reference_halfedge_mesh)) {
                return false;
            }
            weight_mesh = reference_halfedge_mesh.get();
        }

        // number of vertices in the mesh
        const auto num_vertices = static_cast<int>(halfedge_mesh->n_vertices());
        Eigen::VectorXd b_x(num_vertices), b_y(num_vertices), b_z(num_vertices);
        b_x.setZero();
        b_y.setZero();
        b_z.setZero();
        Eigen::SparseMatrix<double> A(num_vertices, num_vertices);

        const auto weight_strategy =
            make_weight_strategy(weight_type, normalize_weights);

        // build
        build_linear_system(
            *halfedge_mesh,
            *weight_mesh,
            A,
            b_x,
            b_y,
            b_z,
            weight_strategy);

        // solve
        const Eigen::VectorXd x = solve_linear_system(A, b_x);
        const Eigen::VectorXd y = solve_linear_system(A, b_y);
        const Eigen::VectorXd z = solve_linear_system(A, b_z);

        // update vertex positions
        update_vertex_positions(*halfedge_mesh, x, y, z);

        /* ----------------------------- Postprocess ------------------------------
        ** Convert the minimal surface mesh from the halfedge structure back to
        ** Geometry format as the node's output.
        */
        auto geometry = openmesh_to_operand(halfedge_mesh.get());

        // Set the output of the nodes
        params.set_output("Output", std::move(*geometry));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

NODE_DECLARATION_UI(hw5_param);
NODE_DEF_CLOSE_SCOPE