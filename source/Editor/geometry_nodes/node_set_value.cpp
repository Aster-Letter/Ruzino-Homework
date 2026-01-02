#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "fem_bem/Expression.hpp"
#include "nodes/core/def/node_def.hpp"


using namespace Ruzino;
using namespace Ruzino::fem_bem;

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(set_value)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<std::string>("Expression").default_val("x + y + z");
    b.add_input<std::string>("Result Name").default_val("result");
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(set_value)
{
    // 获取输入参数
    Geometry input_geometry = params.get_input<Geometry>("Geometry");

    input_geometry.apply_transform();
    std::string expression_str =
        params.get_input<std::string>("Expression").c_str();
    std::string result_name = params.get_input<std::string>("Result Name");

    // 获取网格组件
    auto mesh_component = input_geometry.get_component<MeshComponent>();
    if (!mesh_component) {
        return false;
    }

    // 获取顶点数据
    std::vector<glm::vec3> vertices = mesh_component->get_vertices();
    if (vertices.empty()) {
        return false;
    }

    // 创建表达式
    Expression expr;
    try {
        expr = Expression::from_string(expression_str);
    }
    catch (const std::exception& e) {
        spdlog::error("Error parsing expression: {}", e.what());
        return false;
    }

    // 计算每个顶点的值
    std::vector<float> values;
    values.reserve(vertices.size());

    ParameterMap<float> params_map;

    for (const auto& vertex : vertices) {
        // 应用变换后的坐标
        glm::vec3 transformed_vertex = vertex;

        // 设置参数
        params_map.clear();
        params_map.insert_unchecked("x", transformed_vertex.x);
        params_map.insert_unchecked("y", transformed_vertex.y);
        params_map.insert_unchecked("z", transformed_vertex.z);

        try {
            float value = expr.evaluate_at(params_map);
            values.push_back(value);
        }
        catch (const std::exception& e) {
            spdlog::error(
                "Error evaluating expression at vertex: {}", e.what());
            values.push_back(0.0f);
        }
    }

    // 将结果添加到网格组件
    mesh_component->add_vertex_scalar_quantity(result_name, values);

    // 输出结果
    params.set_output("Geometry", input_geometry);
    return true;
}

NODE_DECLARATION_UI(set_value);

NODE_DEF_CLOSE_SCOPE