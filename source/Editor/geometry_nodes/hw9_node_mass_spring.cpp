#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <glm/geometric.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_set>

#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "mass_spring/FastMassSpring.h"
#include "mass_spring/MassSpring.h"
#include "mass_spring/utils.h"

struct MassSpringStorage {
    constexpr static bool has_storage = false;
    std::shared_ptr<USTC_CG::mass_spring::MassSpring> mass_spring;
    size_t input_topology_hash = 0;
    int solver_type = -1;
    int balanced_shear_enabled = -1;
    int bending_enabled = -1;
    float bending_stiffness_scale = -1.0f;
    int fixed_point_mode = -1;
    double previous_time = 0.0;
};

namespace {
size_t combine_hash(size_t seed, size_t value)
{
    return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) +
                   (seed >> 2));
}

size_t mesh_topology_hash(
    const std::vector<int>& face_counts,
    const std::vector<int>& face_indices)
{
    size_t seed = 1469598103934665603ull;
    seed = combine_hash(seed, face_counts.size());
    for (int count : face_counts) {
        seed = combine_hash(seed, static_cast<size_t>(count));
    }
    seed = combine_hash(seed, face_indices.size());
    for (int index : face_indices) {
        seed = combine_hash(seed, static_cast<size_t>(index));
    }
    return seed;
}

std::vector<glm::vec3> recompute_face_varying_normals(
    const std::vector<glm::vec3>& vertices,
    const std::vector<int>& face_counts,
    const std::vector<int>& face_indices)
{
    std::vector<glm::vec3> normals;
    normals.reserve(face_indices.size());

    int idx = 0;
    for (int face_count : face_counts) {
        if (idx + face_count > static_cast<int>(face_indices.size()) ||
            face_count < 3) {
            break;
        }

        const int i0 = face_indices[idx];
        const int i1 = face_indices[idx + 1];
        const int i2 = face_indices[idx + 2];
        glm::vec3 normal(0.0f, 0.0f, 1.0f);
        if (i0 >= 0 && i1 >= 0 && i2 >= 0 &&
            i0 < static_cast<int>(vertices.size()) &&
            i1 < static_cast<int>(vertices.size()) &&
            i2 < static_cast<int>(vertices.size())) {
            const glm::vec3 edge1 = vertices[i1] - vertices[i0];
            const glm::vec3 edge2 = vertices[i2] - vertices[i0];
            normal = glm::cross(edge1, edge2);
            const float length = glm::length(normal);
            if (length > 1e-8f) {
                normal /= length;
            }
            else {
                normal = glm::vec3(0.0f, 0.0f, 1.0f);
            }
        }

        for (int i = 0; i < face_count; ++i) {
            normals.push_back(normal);
        }
        idx += face_count;
    }

    return normals;
}
}  // namespace

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(hw9_mass_spring)
{
    b.add_input<Geometry>("Mesh");

    // Core simulation parameters shared by all integrators.
    b.add_input<float>("stiffness").default_val(1000).min(100).max(10000);
    b.add_input<float>("h").default_val(0.0333333333f).min(0.0).max(0.5);
    b.add_input<float>("mass").default_val(1.0f).min(0.01f).max(100.0f);
    b.add_input<float>("damping").default_val(0.995).min(0.0).max(1.0);
    b.add_input<float>("gravity").default_val(-9.8).min(-20.).max(20.);

    // Optional sphere collider. The solver treats collision as an explicit
    // linear penalty + normal damping force. Higher penalty values reduce
    // penetration but stiffen the system; damping reduces inward normal speed.
    b.add_input<float>("collision penalty_k")
        .default_val(10000)
        .min(100)
        .max(100000);
    b.add_input<float>("collision damping")
        .default_val(50)
        .min(0)
        .max(10000);
    b.add_input<float>("collision scale factor")
        .default_val(1.1)
        .min(1.0)
        .max(2.0);
    b.add_input<float>("sphere radius").default_val(0.4).min(0.0).max(5.0);
    ;
    b.add_input<float>("sphere center x").default_val(0.0).min(-5.0).max(5.0);
    b.add_input<float>("sphere center y").default_val(0.0).min(-5.0).max(5.0);
    b.add_input<float>("sphere center z").default_val(0.0).min(-5.0).max(5.0);
    // -----------------------------------------------------------------------------------------------------------

    // Solver switches: 0 for implicit Euler, 1 for semi-implicit Euler.
    b.add_input<int>("time integrator type")
        .default_val(0)
        .min(0)
        .max(1);
    b.add_input<int>("enable time profiling").default_val(0).min(0).max(1);
    b.add_input<int>("enable damping").default_val(0).min(0).max(1);
    b.add_input<int>("enable debug output").default_val(0).min(0).max(1);
    b.add_input<int>("implicit max iter").default_val(5).min(1).max(20);
    b.add_input<int>("enable balanced shear springs")
        .default_val(1)
        .min(0)
        .max(1);
    b.add_input<int>("enable bending springs").default_val(1).min(0).max(1);
    b.add_input<float>("bending stiffness scale")
        .default_val(0.05f)
        .min(0.0f)
        .max(1.0f);
    b.add_input<int>("fixed point mode").default_val(0).min(0).max(4);
    b.add_input<int>("semi implicit substeps").default_val(0).min(0).max(200);

    // Optional assignment features.
    b.add_input<int>("enable Liu13").default_val(0).min(0).max(1);
    b.add_input<int>("Liu13 max iter").default_val(20).min(1).max(200);
    b.add_input<int>("enable sphere collision").default_val(0).min(0).max(1);

    // Output
    b.add_output<Geometry>("Output Mesh");
}

NODE_EXECUTION_FUNCTION(hw9_mass_spring)
{
    using namespace Eigen;
    using namespace USTC_CG::mass_spring;

    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto current_time = global_payload.current_time;

    auto& storage = params.get_storage<MassSpringStorage&>();
    auto& mass_spring = storage.mass_spring;

    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    if (mesh->get_face_vertex_counts().size() == 0) {
        throw std::runtime_error("Read USD error.");
    }

    auto apply_runtime_parameters =
        [&](const std::shared_ptr<MassSpring>& solver) {
            solver->stiffness = params.get_input<float>("stiffness");
            solver->h = params.get_input<float>("h");
            solver->mass = params.get_input<float>("mass");
            solver->gravity = { 0, 0, params.get_input<float>("gravity") };
            solver->damping = params.get_input<float>("damping");
            solver->implicit_newton_iterations =
                params.get_input<int>("implicit max iter");
            solver->semi_implicit_substeps =
                params.get_input<int>("semi implicit substeps");
            solver->collision_penalty_k =
                params.get_input<float>("collision penalty_k");
            solver->collision_damping =
                params.get_input<float>("collision damping");
            solver->collision_scale_factor =
                params.get_input<float>("collision scale factor");
            solver->sphere_center = {
                params.get_input<float>("sphere center x"),
                params.get_input<float>("sphere center y"),
                params.get_input<float>("sphere center z")
            };
            solver->sphere_radius = params.get_input<float>("sphere radius");
            solver->enable_sphere_collision =
                params.get_input<int>("enable sphere collision") == 1;
            solver->enable_damping =
                params.get_input<int>("enable damping") == 1;
            solver->time_integrator =
                params.get_input<int>("time integrator type") == 0
                    ? MassSpring::IMPLICIT_EULER
                    : MassSpring::SEMI_IMPLICIT_EULER;
            solver->enable_time_profiling =
                params.get_input<int>("enable time profiling") == 1;
            solver->enable_debug_output =
                params.get_input<int>("enable debug output") == 1;

            if (auto fast_solver =
                    std::dynamic_pointer_cast<FastMassSpring>(solver)) {
                fast_solver->max_iter = params.get_input<int>("Liu13 max iter");
            }
        };

    auto build_fixed_point_mask =
        [&](const Eigen::MatrixXd& vertices) -> std::vector<bool> {
        std::vector<bool> mask(vertices.rows(), false);
        const int mode = params.get_input<int>("fixed point mode");
        if (vertices.rows() == 0) {
            return mask;
        }

        // Mode 0 and 1 both pin the y=max side by coordinates. The original
        // constructor default assumed a square grid and selected vertices
        // [0, sqrt(n)-1], which breaks as soon as the input mesh is rectangular
        // or stores vertices in a different order.
        // Mode 3/4 pin the whole y=max/y=min edge. This is useful for
        // separating a boundary-condition issue from a two-corner cloth setup.
        const bool pin_min_y = mode == 2 || mode == 4;
        const bool pin_full_edge = mode == 3 || mode == 4;
        const double target_y = pin_min_y ? vertices.col(1).minCoeff()
                                          : vertices.col(1).maxCoeff();
        const double y_range =
            vertices.col(1).maxCoeff() - vertices.col(1).minCoeff();
        const double y_eps = std::max(1e-6, 0.02 * std::abs(y_range));

        int left = -1;
        int right = -1;
        int fixed_count = 0;
        double min_x = std::numeric_limits<double>::max();
        double max_x = -std::numeric_limits<double>::max();
        for (int i = 0; i < vertices.rows(); ++i) {
            if (std::abs(vertices(i, 1) - target_y) > y_eps) {
                continue;
            }
            if (pin_full_edge) {
                mask[i] = true;
                ++fixed_count;
                continue;
            }
            if (vertices(i, 0) < min_x) {
                min_x = vertices(i, 0);
                left = i;
            }
            if (vertices(i, 0) > max_x) {
                max_x = vertices(i, 0);
                right = i;
            }
        }

        if (left >= 0) {
            mask[left] = true;
            ++fixed_count;
        }
        if (right >= 0) {
            if (!mask[right]) {
                ++fixed_count;
            }
            mask[right] = true;
        }
        if (params.get_input<int>("enable debug output") == 1) {
            std::cout << "Mass Spring: fixed point mode " << mode
                      << ", fixed vertices = " << left << ", " << right
                      << ", fixed count = " << fixed_count
                      << std::endl;
        }
        return mask;
    };

    std::cout << "Mass Spring: current time = " << current_time << std::endl;
    const double current_time_value = current_time.GetValue();
    const int requested_solver_type = params.get_input<int>("enable Liu13");
    const int requested_balanced_shear =
        params.get_input<int>("enable balanced shear springs");
    const int requested_bending =
        params.get_input<int>("enable bending springs");
    const float requested_bending_scale =
        params.get_input<float>("bending stiffness scale");
    const int requested_fixed_mode = params.get_input<int>("fixed point mode");
    const size_t input_topology_hash =
        mesh ? mesh_topology_hash(
                   mesh->get_face_vertex_counts(),
                   mesh->get_face_vertex_indices())
             : 0;
    const bool simulation_rewound =
        current_time_value < storage.previous_time;
    const bool solver_settings_changed =
        storage.solver_type != requested_solver_type ||
        storage.balanced_shear_enabled != requested_balanced_shear ||
        storage.bending_enabled != requested_bending ||
        storage.bending_stiffness_scale != requested_bending_scale ||
        storage.fixed_point_mode != requested_fixed_mode ||
        storage.input_topology_hash != input_topology_hash;
    storage.previous_time = current_time_value;

    if (current_time_value == 0.0 || !mass_spring || simulation_rewound ||
        solver_settings_changed) {  // Reset and initialize the mass spring class
        if (mesh) {
            if (mass_spring != nullptr)
                mass_spring.reset();

            auto faces = usd_faces_to_eigen(
                mesh->get_face_vertex_counts(),
                mesh->get_face_vertex_indices());
            auto vertices = usd_vertices_to_eigen(mesh->get_vertices());
            auto edges = get_edges(faces);
            EdgeWeightMap edge_stiffness_scales;
            if (params.get_input<int>("enable balanced shear springs") == 1) {
                const auto base_edge_count = edges.size();
                edges = add_balanced_shear_edges(edges, faces, vertices);
                if (params.get_input<int>("enable debug output") == 1) {
                    std::cout << "Mass Spring: balanced shear edges added = "
                              << (edges.size() - base_edge_count)
                              << std::endl;
                }
            }
            if (params.get_input<int>("enable bending springs") == 1) {
                const auto pre_bending_edge_count = edges.size();
                edges = add_bending_edges(
                    edges,
                    faces,
                    vertices,
                    params.get_input<float>("bending stiffness scale"),
                    edge_stiffness_scales);
                if (params.get_input<int>("enable debug output") == 1) {
                    std::cout << "Mass Spring: bending edges added = "
                              << (edges.size() - pre_bending_edge_count)
                              << ", scale = "
                              << params.get_input<float>(
                                     "bending stiffness scale")
                              << std::endl;
                }
            }
            const float k = params.get_input<float>("stiffness");
            const float h = params.get_input<float>("h");

            bool enable_liu13 =
                params.get_input<int>("enable Liu13") == 1 ? true : false;
            if (enable_liu13) {
                // HW Optional
                auto fast_mass_spring =
                    std::make_shared<FastMassSpring>(vertices, edges, k, h);
                fast_mass_spring->max_iter =
                    params.get_input<int>("Liu13 max iter");
                mass_spring = fast_mass_spring;
            }
            else
                mass_spring = std::make_shared<MassSpring>(vertices, edges);

            auto fixed_mask = build_fixed_point_mask(vertices);
            if (!fixed_mask.empty()) {
                mass_spring->set_dirichlet_bc_mask(fixed_mask);
            }
            mass_spring->set_edge_stiffness_scales(edge_stiffness_scales);
            storage.input_topology_hash = input_topology_hash;
            storage.solver_type = requested_solver_type;
            storage.balanced_shear_enabled = requested_balanced_shear;
            storage.bending_enabled = requested_bending;
            storage.bending_stiffness_scale = requested_bending_scale;
            storage.fixed_point_mode = requested_fixed_mode;

            // simulation parameters
            mass_spring->stiffness = k;
            mass_spring->h = params.get_input<float>("h");
            mass_spring->mass = params.get_input<float>("mass");
            mass_spring->gravity = { 0, 0, params.get_input<float>("gravity") };
            mass_spring->damping = params.get_input<float>("damping");
            mass_spring->implicit_newton_iterations =
                params.get_input<int>("implicit max iter");
            mass_spring->semi_implicit_substeps =
                params.get_input<int>("semi implicit substeps");

            // Optional parameters
            // --------- HW Optional: if you implement sphere collision, please
            // uncomment the following lines ------------
            mass_spring->collision_penalty_k =
                params.get_input<float>("collision penalty_k");
            mass_spring->collision_damping =
                params.get_input<float>("collision damping");
            mass_spring->collision_scale_factor =
                params.get_input<float>("collision scale factor");
            float c[3];
            c[0] = params.get_input<float>("sphere center x");
            c[1] = params.get_input<float>("sphere center y");
            c[2] = params.get_input<float>("sphere center z");
            mass_spring->sphere_center = { c[0], c[1], c[2] };
            mass_spring->sphere_radius =
                params.get_input<float>("sphere radius");
            // --------------------------------------------------------------------------------------------------------

            mass_spring->enable_sphere_collision =
                params.get_input<int>("enable sphere collision") == 1 ? true
                                                                      : false;
            mass_spring->enable_damping =
                params.get_input<int>("enable damping") == 1 ? true : false;
            mass_spring->time_integrator =
                params.get_input<int>("time integrator type") == 0
                    ? MassSpring::IMPLICIT_EULER
                    : MassSpring::SEMI_IMPLICIT_EULER;
            mass_spring->enable_time_profiling =
                params.get_input<int>("enable time profiling") == 1 ? true
                                                                    : false;
            mass_spring->enable_debug_output =
                params.get_input<int>("enable debug output") == 1 ? true
                                                                  : false;
        }
        else {
            mass_spring = nullptr;
            throw std::runtime_error("Mass Spring: Need Geometry Input.");
        }
    }
    else if (mass_spring)  // otherwise, step forward the simulation
    {
        apply_runtime_parameters(mass_spring);
        mass_spring->step();
    }
    if (mass_spring) {
        const auto updated_vertices =
            eigen_to_usd_vertices(mass_spring->getX());
        mesh->set_vertices(updated_vertices);

        // The input USD grid carries face-varying normals from the rest pose.
        // After cloth deformation those stale normals make textured cloth
        // look almost black or like a rigid dark sheet. Recompute flat
        // face-varying normals from the deformed positions before output.
        const auto normals = recompute_face_varying_normals(
            updated_vertices,
            mesh->get_face_vertex_counts(),
            mesh->get_face_vertex_indices());
        if (!normals.empty()) {
            mesh->set_normals(normals);
        }

        if (params.get_input<int>("enable debug output") == 1) {
            const Eigen::MatrixXd positions = mass_spring->getX();
            const Eigen::MatrixXd rest_positions = mass_spring->getInitX();
            const Eigen::MatrixXd velocities = mass_spring->getVelocity();
            std::vector<float> velocity_norm;
            std::vector<float> displacement_norm;
            std::vector<float> fixed_mask_quantity;
            velocity_norm.reserve(positions.rows());
            displacement_norm.reserve(positions.rows());
            fixed_mask_quantity.reserve(positions.rows());

            const auto& fixed_mask = mass_spring->getDirichletMask();
            for (int i = 0; i < positions.rows(); ++i) {
                velocity_norm.push_back(
                    static_cast<float>(velocities.row(i).norm()));
                displacement_norm.push_back(
                    static_cast<float>(
                        (positions.row(i) - rest_positions.row(i)).norm()));
                fixed_mask_quantity.push_back(
                    i < static_cast<int>(fixed_mask.size()) && fixed_mask[i]
                        ? 1.0f
                        : 0.0f);
            }
            mesh->add_vertex_scalar_quantity(
                "hw9 velocity norm", velocity_norm);
            mesh->add_vertex_scalar_quantity(
                "hw9 displacement norm", displacement_norm);
            mesh->add_vertex_scalar_quantity(
                "hw9 fixed mask", fixed_mask_quantity);
        }
    }
    params.set_output("Output Mesh", geometry);
    return true;
}

NODE_DECLARATION_UI(hw9_mass_spring);
NODE_DEF_CLOSE_SCOPE
