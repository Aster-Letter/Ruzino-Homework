#include <Eigen/Sparse>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"

/*
** @brief HW4_TutteParameterization
**
** This file contains two nodes whose primary function is to map the boundary of
*a mesh to a plain
** convex closed curve (circle of square), setting the stage for subsequent
*Laplacian equation
** solution and mesh parameterization tasks.
**
** Key to this node's implementation is the adept manipulation of half-edge data
*structures
** to identify and modify the boundary of the mesh.
**
** Task Overview:
** - The two execution functions (node_map_boundary_to_square_exec,
** node_map_boundary_to_circle_exec) require an update to accurately map the
*mesh boundary to a and
** circles. This entails identifying the boundary edges, evenly distributing
*boundary vertices along
** the square's perimeter, and ensuring the internal vertices' positions remain
*unchanged.
** - A focus on half-edge data structures to efficiently traverse and modify
*mesh boundaries.
*/

namespace {

using MeshType = OpenMesh::PolyMesh_ArrayKernelT<>;

constexpr float kPi = 3.14159265358979323846f;

std::vector<OpenMesh::VertexHandle> collect_boundary_loop(MeshType* mesh)
{
    auto start_he = OpenMesh::HalfedgeHandle(-1);
    for (auto he : mesh->halfedges()) {
        if (mesh->is_boundary(he)) {
            start_he = he;
            break;
        }
    }

    if (!start_he.is_valid()) {
        throw std::runtime_error("Boundary Mapping: Mesh has no boundary loop.");
    }

    std::vector<OpenMesh::VertexHandle> boundary_vertices;
    auto current_he = start_he;
    do {
        boundary_vertices.push_back(mesh->to_vertex_handle(current_he));
        current_he = mesh->next_halfedge_handle(current_he);
    } while (current_he.is_valid() && current_he != start_he);

    if (!current_he.is_valid() || boundary_vertices.size() < 2) {
        throw std::runtime_error(
            "Boundary Mapping: Failed to extract a valid boundary loop.");
    }

    return boundary_vertices;
}

std::vector<float> compute_normalized_boundary_arc_lengths(
    MeshType* mesh,
    const std::vector<OpenMesh::VertexHandle>& boundary_vertices)
{
    std::vector<float> cumulative_lengths(boundary_vertices.size(), 0.0f);
    auto total_length = 0.0f;

    for (size_t i = 1; i < boundary_vertices.size(); ++i) {
        total_length += OpenMesh::norm(
            mesh->point(boundary_vertices[i]) -
            mesh->point(boundary_vertices[i - 1]));
        cumulative_lengths[i] = total_length;
    }

    total_length += OpenMesh::norm(
        mesh->point(boundary_vertices.front()) -
        mesh->point(boundary_vertices.back()));

    if (total_length <= 0.0f) {
        throw std::runtime_error(
            "Boundary Mapping: Boundary loop length must be positive.");
    }

    for (auto& length : cumulative_lengths) {
        length /= total_length;
    }

    return cumulative_lengths;
}

OpenMesh::Vec3f square_point_from_unit_perimeter(float t)
{
    if (t < 0.25f) {
        return OpenMesh::Vec3f(t * 4.0f, 0.0f, 0.0f);
    }
    if (t < 0.5f) {
        return OpenMesh::Vec3f(1.0f, (t - 0.25f) * 4.0f, 0.0f);
    }
    if (t < 0.75f) {
        return OpenMesh::Vec3f(1.0f - (t - 0.5f) * 4.0f, 1.0f, 0.0f);
    }

    return OpenMesh::Vec3f(0.0f, 1.0f - (t - 0.75f) * 4.0f, 0.0f);
}

void map_boundary_to_circle(
    MeshType* mesh,
    const std::vector<OpenMesh::VertexHandle>& boundary_vertices)
{
    const auto arc_lengths =
        compute_normalized_boundary_arc_lengths(mesh, boundary_vertices);

    for (size_t i = 0; i < boundary_vertices.size(); ++i) {
        const auto theta = 2.0f * kPi * arc_lengths[i];
        mesh->set_point(
            boundary_vertices[i],
            OpenMesh::Vec3f(
                0.5f * std::cos(theta) + 0.5f,
                0.5f * std::sin(theta) + 0.5f,
                0.0f));
    }
}

void map_boundary_to_square(
    MeshType* mesh,
    const std::vector<OpenMesh::VertexHandle>& boundary_vertices)
{
    const auto arc_lengths =
        compute_normalized_boundary_arc_lengths(mesh, boundary_vertices);

    for (size_t i = 0; i < boundary_vertices.size(); ++i) {
        mesh->set_point(
            boundary_vertices[i],
            square_point_from_unit_perimeter(arc_lengths[i]));
    }
}

} // namespace

NODE_DEF_OPEN_SCOPE

/*
** HW4_TODO: Node to map the mesh boundary to a circle.
*/

NODE_DECLARATION_FUNCTION(hw5_circle_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");
    // Output-1: Processed 3D mesh whose boundary is mapped to a square and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_circle_boundary_mapping)
{
    try {
        // Get the input from params
        auto input = params.get_input<Geometry>("Input");

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
        const auto boundary_vertices = collect_boundary_loop(halfedge_mesh.get());
        map_boundary_to_circle(halfedge_mesh.get(), boundary_vertices);

        /* ----------------------------- Postprocess ------------------------------
        ** Convert the result mesh from the halfedge structure back to Geometry
        *format as the node's
        ** output.
        */
        auto geometry = openmesh_to_operand(halfedge_mesh.get());

        // Set the output of the nodes
        params.set_output("Output", std::move(*geometry));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

/*
** HW4_TODO: Node to map the mesh boundary to a square.
*/

NODE_DECLARATION_FUNCTION(hw5_square_boundary_mapping)
{
    // Input-1: Original 3D mesh with boundary
    b.add_input<Geometry>("Input");

    // Output-1: Processed 3D mesh whose boundary is mapped to a square and the
    // interior vertices remains the same
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_square_boundary_mapping)
{
    try {
        // Get the input from params
        auto input = params.get_input<Geometry>("Input");

        if (!input.get_component<MeshComponent>()) {
            return false;
        }

        /* ----------------------------- Preprocess -------------------------------
        ** Create a halfedge structure (using OpenMesh) for the input mesh.
        */
        auto halfedge_mesh = operand_to_openmesh(&input);
        const auto boundary_vertices = collect_boundary_loop(halfedge_mesh.get());
        map_boundary_to_square(halfedge_mesh.get(), boundary_vertices);


        /* ----------------------------- Postprocess ------------------------------
        ** Convert the result mesh from the halfedge structure back to Geometry
        *format as the node's
        ** output.
        */
        auto geometry = openmesh_to_operand(halfedge_mesh.get());

        // Set the output of the nodes
        params.set_output("Output", std::move(*geometry));
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

NODE_DECLARATION_UI(boundary_mapping);
NODE_DEF_CLOSE_SCOPE