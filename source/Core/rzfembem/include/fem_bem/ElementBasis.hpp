#pragma once
#include <memory>
#include <string>
#include <vector>

#include "Expression.hpp"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec3d.h"

namespace Ruzino {

namespace fem_bem {

    enum class ElementBasisType { FiniteElement, BoundaryElement };

    // Element basis with expression and gradient management
    class RZFEMBEM_API ElementBasis {
       public:
        ElementBasis(unsigned problem_dim, ElementBasisType type);

        // Basic information
        unsigned get_problem_dimension() const;
        unsigned get_element_dimension() const;
        ElementBasisType get_type() const;

        // Expression management
        // Vertex expressions (always available) - 0D knots
        void add_vertex_expression(const std::string& expr_str);
        void set_vertex_expressions(const std::vector<const char*>& expr_strs);
        const std::vector<Expression>& get_vertex_expressions() const;
        void clear_vertex_expressions();
        std::vector<std::vector<Expression>> get_vertex_gradients() const;

        void add_edge_expression(const std::string& expr_str);
        void set_edge_expressions(const std::vector<const char*>& expr_strs);
        const std::vector<Expression>& get_edge_expressions() const;
        void clear_edge_expressions();
        std::vector<std::vector<Expression>> get_edge_gradients() const;

        void add_face_expression(const std::string& expr_str);
        void set_face_expressions(const std::vector<const char*>& expr_strs);
        const std::vector<Expression>& get_face_expressions() const;
        void clear_face_expressions();
        std::vector<std::vector<Expression>> get_face_gradients() const;

        void add_volume_expression(const std::string& expr_str);
        void set_volume_expressions(const std::vector<const char*>& expr_strs);
        const std::vector<Expression>& get_volume_expressions() const;
        void clear_volume_expressions();
        std::vector<std::vector<Expression>> get_volume_gradients() const;

        // Check if specific expression types are available
        bool has_edge_expressions() const;
        bool has_face_expressions() const;
        bool has_volume_expressions() const;

        // Linear mapping interface for coordinate transformation
        ParameterMap<Expression> create_coordinate_mapping(
            const std::vector<pxr::GfVec2d>& world_vertices) const;

        ParameterMap<Expression> create_coordinate_mapping(
            const std::vector<pxr::GfVec3d>& world_vertices) const;

        // Get barycentric variable names
        const std::vector<const char*>& get_barycentric_names() const;

       private:
        // Create a new expression with all necessary variables pre-registered
        Expression create_expression(const std::string& expr_str) const;

        // Setup barycentric coordinate variables based on element dimension
        void setup_barycentric_variables();

       protected:
        unsigned problem_dimension_;
        unsigned element_dimension_;
        ElementBasisType type_;

        std::vector<const char*> barycentric_names_;

        // Expression storage
        std::vector<Expression> vertex_expressions_;
        std::vector<Expression> edge_expressions_;
        std::vector<Expression> face_expressions_;
        std::vector<Expression> volume_expressions_;
    };

    using ElementBasisHandle = std::shared_ptr<ElementBasis>;

    // Factory functions for creating ElementBasisHandle
    RZFEMBEM_API ElementBasisHandle make_fem_1d();
    RZFEMBEM_API ElementBasisHandle make_fem_2d();
    RZFEMBEM_API ElementBasisHandle make_fem_3d();
    RZFEMBEM_API ElementBasisHandle make_bem_2d();
    RZFEMBEM_API ElementBasisHandle make_bem_3d();

}  // namespace fem_bem

}  // namespace Ruzino
