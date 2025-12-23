
#include "GCore/Components/PointsComponent.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"
#include "nvrhi/nvrhi.h"
#include "render_node_base.h"

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(points_step)
{
    b.add_input<nvrhi::BufferHandle>("Points Buffer");
    b.add_input<nvrhi::BufferHandle>("Contact ID Pairs");
    b.add_output<Geometry>("Points");
}

NODE_EXECUTION_FUNCTION(points_step)
{
    auto points_buffer = params.get_input<nvrhi::BufferHandle>("Points Buffer");
    auto contact_id_pairs =
        params.get_input<nvrhi::BufferHandle>("Contact ID Pairs");

    // Placeholder for actual SPH stepping logic
    // In a real implementation, GPU compute shaders would be used here

    Geometry points_geometry = Geometry();

    auto points_component = std::make_shared<PointsComponent>(&points_geometry);
    points_geometry.attach_component(points_component);

    // For demonstration, we just return an empty geometry
    params.set_output("Points", points_geometry);
}

NODE_DECLARATION_UI(points_step);
NODE_DEF_CLOSE_SCOPE
