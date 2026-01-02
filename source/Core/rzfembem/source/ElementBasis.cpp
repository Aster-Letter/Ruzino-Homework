#include "fem_bem/ElementBasis.hpp"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace fem_bem {
ElementBasis::ElementBasis(unsigned problem_dim, ElementBasisType type)
    : problem_dimension_(problem_dim),
      element_dimension_(
          type == ElementBasisType::FiniteElement ? problem_dim
                                                  : problem_dim - 1),
      type_(type)
{
    setup_barycentric_variables();
}
unsigned ElementBasis::get_problem_dimension() const
{
    return problem_dimension_;
}
unsigned ElementBasis::get_element_dimension() const
{
    return element_dimension_;
}
ElementBasisType ElementBasis::get_type() const
{
    return type_;
}
void ElementBasis::add_vertex_expression(const std::string& expr_str)
{
    vertex_expressions_.push_back(create_expression(expr_str));
}
void ElementBasis::set_vertex_expressions(
    const std::vector<const char*>& expr_strs)
{
    vertex_expressions_.clear();
    vertex_expressions_.reserve(expr_strs.size());
    for (const auto& expr_str : expr_strs) {
        vertex_expressions_.push_back(create_expression(expr_str));
    }
}
const std::vector<Expression>& ElementBasis::get_vertex_expressions() const
{
    return vertex_expressions_;
}
void ElementBasis::clear_vertex_expressions()
{
    vertex_expressions_.clear();
}
std::vector<std::vector<Expression>> ElementBasis::get_vertex_gradients() const
{
    std::vector<std::vector<Expression>> gradients;
    gradients.reserve(vertex_expressions_.size());

    for (const auto& expr : vertex_expressions_) {
        std::vector<Expression> grad;
        grad.reserve(barycentric_names_.size());

        for (const auto& var : barycentric_names_) {
            grad.push_back(expr.derivative(var));
        }
        gradients.push_back(std::move(grad));
    }
    return gradients;
}
void ElementBasis::add_edge_expression(const std::string& expr_str)
{
    if (element_dimension_ >= 2) {
        edge_expressions_.push_back(create_expression(expr_str));
    }
}
void ElementBasis::set_edge_expressions(
    const std::vector<const char*>& expr_strs)
{
    if (element_dimension_ >= 2) {
        edge_expressions_.clear();
        edge_expressions_.reserve(expr_strs.size());
        for (const auto& expr_str : expr_strs) {
            edge_expressions_.push_back(create_expression(expr_str));
        }
    }
}
const std::vector<Expression>& ElementBasis::get_edge_expressions() const
{
    return edge_expressions_;
}
void ElementBasis::clear_edge_expressions()
{
    if (element_dimension_ >= 2) {
        edge_expressions_.clear();
    }
}
std::vector<std::vector<Expression>> ElementBasis::get_edge_gradients() const
{
    std::vector<std::vector<Expression>> gradients;
    if (element_dimension_ >= 2) {
        gradients.reserve(edge_expressions_.size());

        for (const auto& expr : edge_expressions_) {
            std::vector<Expression> grad;
            grad.reserve(barycentric_names_.size());

            for (const auto& var : barycentric_names_) {
                grad.push_back(expr.derivative(var));
            }
            gradients.push_back(std::move(grad));
        }
    }
    return gradients;
}
void ElementBasis::add_face_expression(const std::string& expr_str)
{
    if (element_dimension_ >= 3) {
        face_expressions_.push_back(create_expression(expr_str));
    }
}
void ElementBasis::set_face_expressions(
    const std::vector<const char*>& expr_strs)
{
    if (element_dimension_ >= 3) {
        face_expressions_.clear();
        face_expressions_.reserve(expr_strs.size());
        for (const auto& expr_str : expr_strs) {
            face_expressions_.push_back(create_expression(expr_str));
        }
    }
}
const std::vector<Expression>& ElementBasis::get_face_expressions() const
{
    return face_expressions_;
}
void ElementBasis::clear_face_expressions()
{
    if (element_dimension_ >= 3) {
        face_expressions_.clear();
    }
}
std::vector<std::vector<Expression>> ElementBasis::get_face_gradients() const
{
    std::vector<std::vector<Expression>> gradients;
    if (element_dimension_ >= 3) {
        gradients.reserve(face_expressions_.size());

        for (const auto& expr : face_expressions_) {
            std::vector<Expression> grad;
            grad.reserve(barycentric_names_.size());

            for (const auto& var : barycentric_names_) {
                grad.push_back(expr.derivative(var));
            }
            gradients.push_back(std::move(grad));
        }
    }
    return gradients;
}
void ElementBasis::add_volume_expression(const std::string& expr_str)
{
    if (element_dimension_ == 3) {
        volume_expressions_.push_back(create_expression(expr_str));
    }
}
void ElementBasis::set_volume_expressions(
    const std::vector<const char*>& expr_strs)
{
    if (element_dimension_ == 3) {
        volume_expressions_.clear();
        volume_expressions_.reserve(expr_strs.size());
        for (const auto& expr_str : expr_strs) {
            volume_expressions_.push_back(create_expression(expr_str));
        }
    }
}
const std::vector<Expression>& ElementBasis::get_volume_expressions() const
{
    return volume_expressions_;
}
void ElementBasis::clear_volume_expressions()
{
    if (element_dimension_ == 3) {
        volume_expressions_.clear();
    }
}
std::vector<std::vector<Expression>> ElementBasis::get_volume_gradients() const
{
    std::vector<std::vector<Expression>> gradients;
    if (element_dimension_ == 3) {
        gradients.reserve(volume_expressions_.size());

        for (const auto& expr : volume_expressions_) {
            std::vector<Expression> grad;
            grad.reserve(barycentric_names_.size());

            for (const auto& var : barycentric_names_) {
                grad.push_back(expr.derivative(var));
            }
            gradients.push_back(std::move(grad));
        }
    }
    return gradients;
}
bool ElementBasis::has_edge_expressions() const
{
    return element_dimension_ >= 2;
}
bool ElementBasis::has_face_expressions() const
{
    return element_dimension_ >= 3;
}
bool ElementBasis::has_volume_expressions() const
{
    return element_dimension_ == 3;
}
ParameterMap<Expression> ElementBasis::create_coordinate_mapping(
    const std::vector<pxr::GfVec2d>& world_vertices) const
{
    return fem_bem::create_coordinate_mapping(
        barycentric_names_, world_vertices);
}
ParameterMap<Expression> ElementBasis::create_coordinate_mapping(
    const std::vector<pxr::GfVec3d>& world_vertices) const
{
    return fem_bem::create_coordinate_mapping(
        barycentric_names_, world_vertices);
}
const std::vector<const char*>& ElementBasis::get_barycentric_names() const
{
    return barycentric_names_;
}
Expression ElementBasis::create_expression(const std::string& expr_str) const
{
    // Create expression with pre-defined variables
    return Expression(expr_str);
}
void ElementBasis::setup_barycentric_variables()
{
    if (element_dimension_ == 1) {
        barycentric_names_ = { "u1" };
    }
    else if (element_dimension_ == 2) {
        barycentric_names_ = { "u1", "u2" };
    }
    else if (element_dimension_ == 3) {
        barycentric_names_ = { "u1", "u2", "u3" };
    }
}
ElementBasisHandle make_fem_1d()
{
    return std::make_shared<ElementBasis>(1, ElementBasisType::FiniteElement);
}
ElementBasisHandle make_fem_2d()
{
    return std::make_shared<ElementBasis>(2, ElementBasisType::FiniteElement);
}
ElementBasisHandle make_fem_3d()
{
    return std::make_shared<ElementBasis>(3, ElementBasisType::FiniteElement);
}
ElementBasisHandle make_bem_2d()
{
    return std::make_shared<ElementBasis>(2, ElementBasisType::BoundaryElement);
}
ElementBasisHandle make_bem_3d()
{
    return std::make_shared<ElementBasis>(3, ElementBasisType::BoundaryElement);
}
}  // namespace fem_bem
RUZINO_NAMESPACE_CLOSE_SCOPE
