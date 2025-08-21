#include <cstring>
#include <fem_bem/Expression.hpp>

namespace USTC_CG {
namespace fem_bem {
    static exprtk::parser<real> parser;

    // Create coordinate mapping expressions that bind world vertex coordinates
    ParameterMap<Expression> create_coordinate_mapping(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec2d>& world_vertices)
    {
        ParameterMap<Expression> coord_mapping;

        // Handle edge cases gracefully
        if (world_vertices.empty()) {
            coord_mapping.insert_or_assign("x", Expression("0"));
            coord_mapping.insert_or_assign("y", Expression("0"));
            return coord_mapping;
        }

        if (world_vertices.size() == 1) {
            // Single point case
            coord_mapping.insert_or_assign(
                "x", Expression::constant(world_vertices[0][0]));
            coord_mapping.insert_or_assign(
                "y", Expression::constant(world_vertices[0][1]));
            return coord_mapping;
        }

        // For BEM2D (1D elements): we have 2 vertices and 1 barycentric
        // coordinate u1 The mapping is: x = (1-u1)*x0 + u1*x1, y = (1-u1)*y0 +
        // u1*y1
        if (barycentric_names.size() == 1 && world_vertices.size() >= 2) {
            // Create x mapping: (1-u1)*x0 + u1*x1
            std::vector<std::string> x_vars = { "u1", "x0", "x1" };
            Expression x_expr("(1 - u1) * x0 + u1 * x1");
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_expr.bind_variables(x_bindings);

            // Create y mapping: (1-u1)*y0 + u1*y1
            std::vector<std::string> y_vars = { "u1", "y0", "y1" };
            Expression y_expr("(1 - u1) * y0 + u1 * y1");
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_expr.bind_variables(y_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            return coord_mapping;
        }

        // For FEM2D (2D elements): we have 3 vertices and 2 barycentric
        // coordinates u1, u2 The mapping is: x = (1-u1-u2)*x0 + u1*x1 + u2*x2
        if (barycentric_names.size() == 2 && world_vertices.size() >= 3) {
            // Create x mapping: (1-u1-u2)*x0 + u1*x1 + u2*x2
            std::vector<std::string> x_vars = { "u1", "u2", "x0", "x1", "x2" };
            Expression x_expr("(1 - u1 - u2) * x0 + u1 * x1 + u2 * x2");
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_bindings.insert_or_assign(
                "x2", static_cast<real>(world_vertices[2][0]));
            x_expr.bind_variables(x_bindings);

            // Create y mapping: (1-u1-u2)*y0 + u1*y1 + u2*y2
            std::vector<std::string> y_vars = { "u1", "u2", "y0", "y1", "y2" };
            Expression y_expr("(1 - u1 - u2) * y0 + u1 * y1 + u2 * y2");
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_bindings.insert_or_assign(
                "y2", static_cast<real>(world_vertices[2][1]));
            y_expr.bind_variables(y_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            return coord_mapping;
        }

        // Fallback for other cases
        std::size_t actual_vertices =
            std::min(world_vertices.size(), barycentric_names.size() + 1);

        // Build the base expression string for x coordinate
        std::string x_base = "(1";
        for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
            x_base += " - " + barycentric_names[i];
        }
        x_base += ") * x0";

        for (std::size_t i = 0;
             i < std::min(barycentric_names.size(), actual_vertices - 1);
             ++i) {
            x_base +=
                " + " + barycentric_names[i] + " * x" + std::to_string(i + 1);
        }

        // Create variable list including all coordinate variables
        std::vector<std::string> x_vars = barycentric_names;
        for (std::size_t i = 0; i < actual_vertices; ++i) {
            x_vars.push_back("x" + std::to_string(i));
        }

        Expression x_expr(x_base);
        ParameterMap<real> x_bindings;
        for (std::size_t i = 0; i < actual_vertices; ++i) {
            x_bindings.insert_or_assign(
                ("x" + std::to_string(i)).c_str(),
                static_cast<real>(world_vertices[i][0]));
        }
        x_expr.bind_variables(x_bindings);

        // Build the base expression string for y coordinate
        std::string y_base = "(1";
        for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
            y_base += " - " + barycentric_names[i];
        }
        y_base += ") * y0";

        for (std::size_t i = 0;
             i < std::min(barycentric_names.size(), actual_vertices - 1);
             ++i) {
            y_base +=
                " + " + barycentric_names[i] + " * y" + std::to_string(i + 1);
        }

        // Create variable list including all coordinate variables
        std::vector<std::string> y_vars = barycentric_names;
        for (std::size_t i = 0; i < actual_vertices; ++i) {
            y_vars.push_back("y" + std::to_string(i));
        }

        Expression y_expr(y_base);
        ParameterMap<real> y_bindings;
        for (std::size_t i = 0; i < actual_vertices; ++i) {
            y_bindings.insert_or_assign(
                ("y" + std::to_string(i)).c_str(),
                static_cast<real>(world_vertices[i][1]));
        }
        y_expr.bind_variables(y_bindings);

        coord_mapping.insert_or_assign("x", x_expr);
        coord_mapping.insert_or_assign("y", y_expr);

        return coord_mapping;
    }

    ParameterMap<Expression> create_coordinate_mapping(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec3d>& world_vertices)
    {
        ParameterMap<Expression> coord_mapping;

        // Handle edge cases gracefully
        if (world_vertices.empty()) {
            coord_mapping.insert_or_assign("x", Expression("0"));
            coord_mapping.insert_or_assign("y", Expression("0"));
            coord_mapping.insert_or_assign("z", Expression("0"));
            return coord_mapping;
        }

        if (world_vertices.size() == 1) {
            coord_mapping.insert_or_assign(
                "x", Expression::constant(world_vertices[0][0]));
            coord_mapping.insert_or_assign(
                "y", Expression::constant(world_vertices[0][1]));
            coord_mapping.insert_or_assign(
                "z", Expression::constant(world_vertices[0][2]));
            return coord_mapping;
        }

        // For BEM2D in 3D (1D elements): we have 2 vertices and 1 barycentric
        // coordinate u1
        if (barycentric_names.size() == 1 && world_vertices.size() >= 2) {
            // Create x mapping: (1-u1)*x0 + u1*x1
            std::vector<std::string> x_vars = { "u1", "x0", "x1" };
            Expression x_expr("(1 - u1) * x0 + u1 * x1");
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_expr.bind_variables(x_bindings);

            // Create y mapping
            std::vector<std::string> y_vars = { "u1", "y0", "y1" };
            Expression y_expr("(1 - u1) * y0 + u1 * y1");
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_expr.bind_variables(y_bindings);

            // Create z mapping
            std::vector<std::string> z_vars = { "u1", "z0", "z1" };
            Expression z_expr("(1 - u1) * z0 + u1 * z1");
            ParameterMap<real> z_bindings;
            z_bindings.insert_or_assign(
                "z0", static_cast<real>(world_vertices[0][2]));
            z_bindings.insert_or_assign(
                "z1", static_cast<real>(world_vertices[1][2]));
            z_expr.bind_variables(z_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            coord_mapping.insert_or_assign("z", z_expr);
            return coord_mapping;
        }

        // For BEM3D (2D elements): we have 3 vertices and 2 barycentric
        // coordinates u1, u2
        if (barycentric_names.size() == 2 && world_vertices.size() >= 3) {
            // Create x mapping: (1-u1-u2)*x0 + u1*x1 + u2*x2
            std::vector<std::string> x_vars = { "u1", "u2", "x0", "x1", "x2" };
            Expression x_expr("(1 - u1 - u2) * x0 + u1 * x1 + u2 * x2");
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_bindings.insert_or_assign(
                "x2", static_cast<real>(world_vertices[2][0]));
            x_expr.bind_variables(x_bindings);

            // Create y mapping
            std::vector<std::string> y_vars = { "u1", "u2", "y0", "y1", "y2" };
            Expression y_expr("(1 - u1 - u2) * y0 + u1 * y1 + u2 * y2");
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_bindings.insert_or_assign(
                "y2", static_cast<real>(world_vertices[2][1]));
            y_expr.bind_variables(y_bindings);

            // Create z mapping
            std::vector<std::string> z_vars = { "u1", "u2", "z0", "z1", "z2" };
            Expression z_expr("(1 - u1 - u2) * z0 + u1 * z1 + u2 * z2");
            ParameterMap<real> z_bindings;
            z_bindings.insert_or_assign(
                "z0", static_cast<real>(world_vertices[0][2]));
            z_bindings.insert_or_assign(
                "z1", static_cast<real>(world_vertices[1][2]));
            z_bindings.insert_or_assign(
                "z2", static_cast<real>(world_vertices[2][2]));
            z_expr.bind_variables(z_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            coord_mapping.insert_or_assign("z", z_expr);
            return coord_mapping;
        }

        // For FEM3D (3D elements): we have 4 vertices and 3 barycentric
        // coordinates u1, u2, u3
        if (barycentric_names.size() == 3 && world_vertices.size() >= 4) {
            // Create x mapping: (1-u1-u2-u3)*x0 + u1*x1 + u2*x2 + u3*x3
            std::vector<std::string> x_vars = { "u1", "u2", "u3", "x0",
                                                "x1", "x2", "x3" };
            Expression x_expr(
                "(1 - u1 - u2 - u3) * x0 + u1 * x1 + u2 * x2 + u3 * x3");
            ParameterMap<real> x_bindings;
            x_bindings.insert_or_assign(
                "x0", static_cast<real>(world_vertices[0][0]));
            x_bindings.insert_or_assign(
                "x1", static_cast<real>(world_vertices[1][0]));
            x_bindings.insert_or_assign(
                "x2", static_cast<real>(world_vertices[2][0]));
            x_bindings.insert_or_assign(
                "x3", static_cast<real>(world_vertices[3][0]));
            x_expr.bind_variables(x_bindings);

            // Similar for y and z
            std::vector<std::string> y_vars = { "u1", "u2", "u3", "y0",
                                                "y1", "y2", "y3" };
            Expression y_expr(
                "(1 - u1 - u2 - u3) * y0 + u1 * y1 + u2 * y2 + u3 * y3");
            ParameterMap<real> y_bindings;
            y_bindings.insert_or_assign(
                "y0", static_cast<real>(world_vertices[0][1]));
            y_bindings.insert_or_assign(
                "y1", static_cast<real>(world_vertices[1][1]));
            y_bindings.insert_or_assign(
                "y2", static_cast<real>(world_vertices[2][1]));
            y_bindings.insert_or_assign(
                "y3", static_cast<real>(world_vertices[3][1]));
            y_expr.bind_variables(y_bindings);

            std::vector<std::string> z_vars = { "u1", "u2", "u3", "z0",
                                                "z1", "z2", "z3" };
            Expression z_expr(
                "(1 - u1 - u2 - u3) * z0 + u1 * z1 + u2 * z2 + u3 * z3");
            ParameterMap<real> z_bindings;
            z_bindings.insert_or_assign(
                "z0", static_cast<real>(world_vertices[0][2]));
            z_bindings.insert_or_assign(
                "z1", static_cast<real>(world_vertices[1][2]));
            z_bindings.insert_or_assign(
                "z2", static_cast<real>(world_vertices[2][2]));
            z_bindings.insert_or_assign(
                "z3", static_cast<real>(world_vertices[3][2]));
            z_expr.bind_variables(z_bindings);

            coord_mapping.insert_or_assign("x", x_expr);
            coord_mapping.insert_or_assign("y", y_expr);
            coord_mapping.insert_or_assign("z", z_expr);
            return coord_mapping;
        }

        // Fallback for other cases
        std::size_t actual_vertices =
            std::min(world_vertices.size(), barycentric_names.size() + 1);

        // Build expressions for x, y, z coordinates
        for (std::size_t coord = 0; coord < 3; ++coord) {
            std::string coord_name = (coord == 0)   ? "x"
                                     : (coord == 1) ? "y"
                                                    : "z";
            std::string var_prefix = coord_name;

            // Build the base expression string
            std::string expr_str = "(1";
            for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
                expr_str += " - " + barycentric_names[i];
            }
            expr_str += ") * " + var_prefix + "0";

            for (std::size_t i = 0;
                 i < std::min(barycentric_names.size(), actual_vertices - 1);
                 ++i) {
                expr_str += " + " + barycentric_names[i] + " * " + var_prefix +
                            std::to_string(i + 1);
            }

            // Create variable list including all coordinate variables
            std::vector<std::string> coord_vars = barycentric_names;
            for (std::size_t i = 0; i < actual_vertices; ++i) {
                coord_vars.push_back(var_prefix + std::to_string(i));
            }

            Expression coord_expr(expr_str);
            ParameterMap<real> coord_bindings;
            for (std::size_t i = 0; i < actual_vertices; ++i) {
                real coord_value =
                    (coord == 0)   ? static_cast<real>(world_vertices[i][0])
                    : (coord == 1) ? static_cast<real>(world_vertices[i][1])
                                   : static_cast<real>(world_vertices[i][2]);
                coord_bindings.insert_or_assign(
                    (var_prefix + std::to_string(i)).c_str(), coord_value);
            }
            coord_expr.bind_variables(coord_bindings);

            coord_mapping.insert_or_assign(coord_name.c_str(), coord_expr);
        }

        return coord_mapping;
    }

    // Helper to create mapped expression using coordinate mapping
    Expression create_mapped_expression_with_coord_mapping(
        const Expression& expr,
        const ParameterMap<Expression>& coord_mapping,
        const std::vector<std::string>& barycentric_names)
    {
        // Create substitution map from coordinate mapping
        ParameterMap<Expression> substitutions;

        for (std::size_t i = 0; i < coord_mapping.size(); ++i) {
            const char* coord_name = coord_mapping.get_name_at(i);
            const Expression& coord_expr = coord_mapping.get_value_at(i);
            substitutions.insert_or_assign(coord_name, coord_expr);
        }

        // Create compound expression that substitutes physical coordinates
        return Expression(expr, substitutions);
    }

}  // namespace fem_bem
}  // namespace USTC_CG
