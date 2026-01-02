#include "nodes/core/def/node_def.hpp"
#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/algorithms/intersection.h"
#include <glm/glm.hpp>

NODE_DEF_OPEN_SCOPE

// Storage for persistent simulation state
struct MassSpringStorage {
    constexpr static bool has_storage = true;
    
    std::vector<glm::vec3> velocities;
    std::vector<glm::vec3> rest_positions;
    std::vector<std::pair<int, int>> springs;
    std::vector<float> rest_lengths;
    bool initialized = false;
};

NODE_DECLARATION_FUNCTION(mass_spring)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<float>("Mass").default_val(1.0f).min(0.01f).max(100.0f);
    b.add_input<float>("Stiffness").default_val(1000.0f).min(1.0f).max(10000.0f);
    b.add_input<float>("Damping").default_val(0.99f).min(0.0f).max(1.0f);
    b.add_input<int>("Substeps").default_val(5).min(1).max(20);
    b.add_input<float>("Gravity").default_val(-9.81f).min(-20.0f).max(0.0f);
    b.add_input<float>("Ground Restitution").default_val(0.3f).min(0.0f).max(1.0f);
    
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(mass_spring)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto& storage = params.get_storage<MassSpringStorage&>();
    
    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();
    
    float mass = params.get_input<float>("Mass");
    float stiffness = params.get_input<float>("Stiffness");
    float damping = params.get_input<float>("Damping");
    int substeps = params.get_input<int>("Substeps");
    float gravity = params.get_input<float>("Gravity");
    float restitution = params.get_input<float>("Ground Restitution");
    float dt = global_payload.delta_time;
    
    // Try to get mesh component first, then points
    auto mesh_component = input_geom.get_component<MeshComponent>();
    std::vector<glm::vec3> positions;
    std::vector<int> face_vertex_indices;
    
    if (mesh_component) {
        positions = mesh_component->get_vertices();
        face_vertex_indices = mesh_component->get_face_vertex_indices();
    } else {
        auto points_component = input_geom.get_component<PointsComponent>();
        if (!points_component) {
            params.set_output<Geometry>("Geometry", std::move(input_geom));
            return true;
        }
        positions = points_component->get_vertices();
    }
    
    int num_particles = positions.size();
    if (num_particles == 0) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }
    
    // Initialize storage on first run
    if (!storage.initialized || storage.velocities.size() != num_particles) {
        storage.velocities.resize(num_particles, glm::vec3(0.0f));
        storage.rest_positions = positions;
        storage.springs.clear();
        storage.rest_lengths.clear();
        
        // Build springs from mesh edges
        if (mesh_component && !face_vertex_indices.empty()) {
            std::set<std::pair<int, int>> edge_set;
            
            // Extract edges from faces
            int idx = 0;
            auto face_counts = mesh_component->get_face_vertex_counts();
            for (int face_count : face_counts) {
                for (int i = 0; i < face_count; ++i) {
                    int v0 = face_vertex_indices[idx + i];
                    int v1 = face_vertex_indices[idx + (i + 1) % face_count];
                    
                    // Store edge with smaller index first
                    if (v0 > v1) std::swap(v0, v1);
                    edge_set.insert({v0, v1});
                }
                idx += face_count;
            }
            
            // Create springs from edges
            for (const auto& edge : edge_set) {
                storage.springs.push_back(edge);
                float rest_len = glm::length(
                    storage.rest_positions[edge.first] - 
                    storage.rest_positions[edge.second]);
                storage.rest_lengths.push_back(rest_len);
            }
        } else {
            // For point cloud, create springs based on proximity
            float search_radius = 0.2f; // Adjustable parameter
            for (int i = 0; i < num_particles; ++i) {
                for (int j = i + 1; j < num_particles; ++j) {
                    float dist = glm::length(positions[i] - positions[j]);
                    if (dist < search_radius) {
                        storage.springs.push_back({i, j});
                        storage.rest_lengths.push_back(dist);
                    }
                }
            }
        }
        
        storage.initialized = true;
    }
    
    // Semi-implicit Euler integration with substeps
    float dt_sub = dt / substeps;
    
    for (int step = 0; step < substeps; ++step) {
        // Compute forces
        std::vector<glm::vec3> forces(num_particles, glm::vec3(0.0f));
        
        // Gravity
        for (int i = 0; i < num_particles; ++i) {
            forces[i] += glm::vec3(0.0f, 0.0f, gravity * mass);
        }
        
        // Spring forces
        for (size_t s = 0; s < storage.springs.size(); ++s) {
            int i = storage.springs[s].first;
            int j = storage.springs[s].second;
            
            glm::vec3 dir = positions[j] - positions[i];
            float current_len = glm::length(dir);
            
            if (current_len > 1e-6f) {
                dir /= current_len;
                float extension = current_len - storage.rest_lengths[s];
                glm::vec3 spring_force = stiffness * extension * dir;
                
                forces[i] += spring_force;
                forces[j] -= spring_force;
            }
        }
        
        // Semi-implicit Euler: update velocity then position
        for (int i = 0; i < num_particles; ++i) {
            glm::vec3 acceleration = forces[i] / mass;
            storage.velocities[i] += acceleration * dt_sub;
            storage.velocities[i] *= damping; // Apply damping
            positions[i] += storage.velocities[i] * dt_sub;
        }
        
        // Ground collision (z = 0 plane)
        for (int i = 0; i < num_particles; ++i) {
            if (positions[i].z < 0.0f) {
                positions[i].z = 0.0f;
                
                // Reflect velocity with restitution
                if (storage.velocities[i].z < 0.0f) {
                    storage.velocities[i].z = -storage.velocities[i].z * restitution;
                    
                    // Apply friction to tangential velocity
                    float friction = 0.8f;
                    storage.velocities[i].x *= friction;
                    storage.velocities[i].y *= friction;
                }
            }
        }
    }
    
    // Update geometry with new positions
    if (mesh_component) {
        mesh_component->set_vertices(positions);
    } else {
        auto points_component = input_geom.get_component<PointsComponent>();
        points_component->set_vertices(positions);
    }
    
    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(mass_spring);
NODE_DEF_CLOSE_SCOPE
