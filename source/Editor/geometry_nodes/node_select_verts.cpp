#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "nodes/core/def/node_def.hpp"

using namespace Ruzino;

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(select_verts)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::string>("Scalar Name").default_val("result");
    b.add_input<float>("Threshold").default_val(0.0f);
    b.add_input<std::string>("Mask Name").default_val("mask");
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(select_verts)
{
    // 获取输入参数
    Geometry input_geometry = params.get_input<Geometry>("Geometry");
    input_geometry.apply_transform();
    std::string scalar_name = params.get_input<std::string>("Scalar Name");
    float threshold = params.get_input<float>("Threshold");
    std::string mask_name = params.get_input<std::string>("Mask Name");

    // 获取网格组件
    auto mesh_component = input_geometry.get_component<MeshComponent>();
    if (!mesh_component) {
        return false;
    }

    // 获取标量数据
    std::vector<float> scalar_data = mesh_component->get_vertex_scalar_quantity(scalar_name);
    if (scalar_data.empty()) {
        return false;
    }

    // 创建mask
    std::vector<float> mask;
    mask.reserve(scalar_data.size());
    
    for (float value : scalar_data) {
        mask.push_back(value > threshold ? 1.0f : 0.0f);
    }

    // 将mask添加到网格组件
    mesh_component->add_vertex_scalar_quantity(mask_name, mask);

    // 输出结果
    params.set_output("Geometry", input_geometry);
    return true;
}

NODE_DECLARATION_UI(select_verts);

NODE_DEF_CLOSE_SCOPE