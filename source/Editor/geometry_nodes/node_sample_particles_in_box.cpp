#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <memory>

#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "sph_fluid/sph_base.h"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(sample_particles_in_box)
{
    b.add_input<float>("sim box min x").default_val(-1).min(-10).max(0);
    b.add_input<float>("sim box min y").default_val(-1).min(-10).max(0);
    b.add_input<float>("sim box min z").default_val(-1).min(-10).max(0);
    b.add_input<float>("sim box max x").default_val(1).min(0).max(10);
    b.add_input<float>("sim box max y").default_val(1).min(0).max(10);
    b.add_input<float>("sim box max z").default_val(1).min(0).max(10);
    b.add_input<int>("num particle x").default_val(10).min(1).max(30);
    b.add_input<int>("num particle y").default_val(10).min(1).max(30);
    b.add_input<int>("num particle z").default_val(10).min(1).max(30);
    b.add_input<float>("point width").default_val(0.05f).min(0.01f).max(0.5f);

    b.add_output<Geometry>("Points");
}

NODE_EXECUTION_FUNCTION(sample_particles_in_box)
{
    using namespace Eigen;
    using namespace USTC_CG::sph_fluid;

    auto particle_box_min = glm::vec3(
        params.get_input<float>("sim box min x"),
        params.get_input<float>("sim box min y"),
        params.get_input<float>("sim box min z")
    );
    auto particle_box_max = glm::vec3(params.get_input<float>("sim box max x"),
        params.get_input<float>("sim box max y"),
        params.get_input<float>("sim box max z")
    );
    auto num_particle_per_axis = std::array<int, 3>{ params.get_input<int>("num particle x"),
                            params.get_input<int>("num particle y"),
                                params.get_input<int>("num particle z")

    };
    // Check the parameters
    if (particle_box_max[0] <= particle_box_min[0] ||
        particle_box_max[1] <= particle_box_min[1] ||
        particle_box_max[2] <= particle_box_min[2]) {
        throw std::runtime_error("Invalid particle samping box.");
    }
    if (num_particle_per_axis[0] <= 0 || num_particle_per_axis[1] <= 0 ||
        num_particle_per_axis[2] <= 0) {
        throw std::runtime_error("Invalid number of particles per axis.");
    }
    // Sample particles in the box
    MatrixXd particle_pos = ParticleSystem::sample_particle_pos_in_a_box(
        { particle_box_min[0], particle_box_min[1], particle_box_min[2] },
        { particle_box_max[0], particle_box_max[1], particle_box_max[2] },
        { num_particle_per_axis[0],
          num_particle_per_axis[1],
          num_particle_per_axis[2] });

    // Construct necessary output
    auto geometry = Geometry();
    auto points_component = std::make_shared<PointsComponent>(&geometry);
    geometry.attach_component(points_component);

    std::vector<glm::vec3> vertices;
    for (int i = 0; i < particle_pos.rows(); i++) {
        vertices.push_back(
            glm::vec3(
                particle_pos(i, 0), particle_pos(i, 1), particle_pos(i, 2)));
    }

    points_component->set_vertices(vertices);

    float point_width = params.get_input<float>("point width");
    points_component->set_width(
        std::vector<float>(vertices.size(), point_width));

    params.set_output("Points", std::move(geometry));
    return true;
}

NODE_DECLARATION_UI(sample_particles_in_box);
NODE_DEF_CLOSE_SCOPE