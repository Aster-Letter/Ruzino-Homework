#pragma once
#include "GCore/GOP.h"

#include <exprtk/exprtk.hpp>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "integrate.hpp"
#include "pxr/base/gf/vec3f.h"
USTC_CG_NAMESPACE_OPEN_SCOPE

namespace fem_bem {

enum class ElementBasisType { FiniteElement, BoundaryElement };

// Base template for element basis with dimension-aware expression storage
template<unsigned ProblemDimension, ElementBasisType Type>
class ElementBasis {
   public:
    static constexpr unsigned problem_dimension = ProblemDimension;
    static constexpr unsigned element_dimension =
        (Type == ElementBasisType::FiniteElement) ? ProblemDimension
                                                  : ProblemDimension - 1;
    static constexpr ElementBasisType type = Type;

    static_assert(
        ProblemDimension >= 1 && ProblemDimension <= 3,
        "Problem dimension must be 1, 2, or 3");
    static_assert(
        Type == ElementBasisType::BoundaryElement ? ProblemDimension >= 2
                                                  : true,
        "Boundary elements require problem dimension >= 2");

    ElementBasis() = default;
    virtual ~ElementBasis() = default;

    // Vertex expressions (always available) - 0D knots
    void add_vertex_expression(const std::string& expr)
    {
        vertex_expressions_.push_back(expr);
    }
    void set_vertex_expressions(const std::vector<std::string>& exprs)
    {
        vertex_expressions_ = exprs;
    }
    const std::vector<std::string>& get_vertex_expressions() const
    {
        return vertex_expressions_;
    }
    void clear_vertex_expressions()
    {
        vertex_expressions_.clear();
    }

    // Edge expressions (available when element_dimension >= 2) - 1D knots
    template<typename T = void>
    std::enable_if_t<element_dimension >= 2, void> add_edge_expression(
        const std::string& expr)
    {
        edge_expressions_.push_back(expr);
    }

    template<typename T = void>
    std::enable_if_t<element_dimension >= 2, void> set_edge_expressions(
        const std::vector<std::string>& exprs)
    {
        edge_expressions_ = exprs;
    }

    template<typename T = void>
    std::enable_if_t<element_dimension >= 2, const std::vector<std::string>&>
    get_edge_expressions() const
    {
        return edge_expressions_;
    }

    template<typename T = void>
    std::enable_if_t<element_dimension >= 2, void> clear_edge_expressions()
    {
        edge_expressions_.clear();
    }

    // Face expressions (available when element_dimension >= 3) - 2D knots
    template<typename T = void>
    std::enable_if_t<element_dimension >= 3, void> add_face_expression(
        const std::string& expr)
    {
        face_expressions_.push_back(expr);
    }

    template<typename T = void>
    std::enable_if_t<element_dimension >= 3, void> set_face_expressions(
        const std::vector<std::string>& exprs)
    {
        face_expressions_ = exprs;
    }

    template<typename T = void>
    std::enable_if_t<element_dimension >= 3, const std::vector<std::string>&>
    get_face_expressions() const
    {
        return face_expressions_;
    }

    template<typename T = void>
    std::enable_if_t<element_dimension >= 3, void> clear_face_expressions()
    {
        face_expressions_.clear();
    }

    // Volume expressions (only for 3D elements) - 3D knots
    template<typename T = void>
    std::enable_if_t<element_dimension == 3, void> add_volume_expression(
        const std::string& expr)
    {
        volume_expressions_.push_back(expr);
    }

    template<typename T = void>
    std::enable_if_t<element_dimension == 3, void> set_volume_expressions(
        const std::vector<std::string>& exprs)
    {
        volume_expressions_ = exprs;
    }

    template<typename T = void>
    std::enable_if_t<element_dimension == 3, const std::vector<std::string>&>
    get_volume_expressions() const
    {
        return volume_expressions_;
    }

    template<typename T = void>
    std::enable_if_t<element_dimension == 3, void> clear_volume_expressions()
    {
        volume_expressions_.clear();
    }

    // Check if specific expression types are available at compile time
    static constexpr bool has_edge_expressions()
    {
        return element_dimension >= 2;
    }
    static constexpr bool has_face_expressions()
    {
        return element_dimension >= 3;
    }
    static constexpr bool has_volume_expressions()
    {
        return element_dimension == 3;
    }

   protected:
    // Vertex expressions are always available (0D knots)
    std::vector<std::string> vertex_expressions_;

    // Conditional member variables based on element dimension
    std::conditional_t<element_dimension >= 2, std::vector<std::string>, char>
        edge_expressions_;
    std::conditional_t<element_dimension >= 3, std::vector<std::string>, char>
        face_expressions_;
    std::conditional_t<element_dimension == 3, std::vector<std::string>, char>
        volume_expressions_;
};

// Convenient type aliases for common cases
template<unsigned ProblemDim>
using FiniteElementBasis =
    ElementBasis<ProblemDim, ElementBasisType::FiniteElement>;

template<unsigned ProblemDim>
using BoundaryElementBasis =
    ElementBasis<ProblemDim, ElementBasisType::BoundaryElement>;

// Specific instantiations with clear documentation
using FEM1D = FiniteElementBasis<1>;  // 1D finite elements (element_dim = 1)
using FEM2D = FiniteElementBasis<2>;  // 2D finite elements (element_dim = 2)
using FEM3D =
    FiniteElementBasis<3>;  // 3D finite elements (element_dim = 3, has volume)

using BEM2D = BoundaryElementBasis<2>;  // 1D boundary elements in 2D space
                                        // (element_dim = 1)
using BEM3D = BoundaryElementBasis<3>;  // 2D boundary elements in 3D space
                                        // (element_dim = 2)

// Base class for polymorphism when needed
class ElementBasisBase {
   public:
    virtual ~ElementBasisBase() = default;
    virtual ElementBasisType get_type() const = 0;
    virtual unsigned get_problem_dimension() const = 0;
    virtual unsigned get_element_dimension() const = 0;

    // Virtual interface for expressions
    virtual void add_vertex_expression(const std::string& expr) = 0;
    virtual const std::vector<std::string>& get_vertex_expressions() const = 0;
};

// Wrapper for type erasure when polymorphism is needed
template<unsigned ProblemDim, ElementBasisType Type>
class ElementBasisWrapper : public ElementBasisBase,
                            public ElementBasis<ProblemDim, Type> {
   public:
    ElementBasisType get_type() const override
    {
        return Type;
    }
    unsigned get_problem_dimension() const override
    {
        return ProblemDim;
    }
    unsigned get_element_dimension() const override
    {
        return ElementBasis<ProblemDim, Type>::element_dimension;
    }

    void add_vertex_expression(const std::string& expr) override
    {
        ElementBasis<ProblemDim, Type>::add_vertex_expression(expr);
    }

    const std::vector<std::string>& get_vertex_expressions() const override
    {
        return ElementBasis<ProblemDim, Type>::get_vertex_expressions();
    }
};

using ElementBasisHandle = std::shared_ptr<ElementBasisBase>;

}  // namespace fem_bem

USTC_CG_NAMESPACE_CLOSE_SCOPE
