#include <fem_bem/Expression.hpp>
namespace USTC_CG {
namespace fem_bem {

    real numerical_derivative(
        const exprtk::expression<real>& expr,
        exprtk::details::variable_node<real>* var,
        const real& h)
    {
        const real x_init = var->ref();

        // Use 2-point central difference with better stability
        var->ref() = x_init + h;
        const real y_plus = expr.value();
        var->ref() = x_init - h;
        const real y_minus = expr.value();
        var->ref() = x_init;

        // Check for numerical issues and use adaptive step if needed
        real derivative = (y_plus - y_minus) / (real(2) * h);

        // If derivative seems too large, try with smaller step
        if (std::abs(derivative) > real(1e6)) {
            real smaller_h = h * real(0.1);
            var->ref() = x_init + smaller_h;
            const real y_plus_small = expr.value();
            var->ref() = x_init - smaller_h;
            const real y_minus_small = expr.value();
            var->ref() = x_init;
            derivative = (y_plus_small - y_minus_small) / (real(2) * smaller_h);
        }

        return derivative;
    }
    std::function<real(const ParameterMap<real>&)> create_derivative_function(
        const std::string& expression_string,
        const std::string& variable_name,
        const real& h)
    {
        return [expression_string, variable_name, h](
                   const ParameterMap<real>& values) -> real {
            // Create a temporary expression for derivative computation
            exprtk::symbol_table<real> symbol_table;
            exprtk::expression<real> expr;
            exprtk::parser<real> parser;

            // Add all variables to symbol table
            ParameterMap<real> temp_values = values;

            for (std::size_t i = 0; i < temp_values.size(); ++i) {
                const char* name = temp_values.get_name_at(i);
                const real& value = temp_values.get_value_at(i);
                symbol_table.add_variable(
                    name, const_cast<real&>(temp_values.get_value_at(i)));
            }

            symbol_table.add_constants();
            expr.register_symbol_table(symbol_table);

            if (!parser.compile(expression_string, expr)) {
                return real(0);  // Return 0 for invalid expressions
            }

            // Find the variable node for differentiation
            auto* var_node = symbol_table.get_variable(variable_name);
            if (!var_node) {
                return real(0);  // Variable not found
            }

            return numerical_derivative(expr, var_node, h);
        };
    }

    // Create a derivative function for compound expressions using chain rule
    std::function<real(const ParameterMap<real>&)>
    create_compound_derivative_function(
        const std::function<real(const ParameterMap<real>&)>&
            compound_evaluator,
        const std::string& variable_name,
        const real& h)
    {
        return [compound_evaluator, variable_name, h](
                   const ParameterMap<real>& values) -> real {
            const real* var_value = values.find(variable_name.c_str());
            if (!var_value) {
                return real(0);  // Variable not found
            }

            const real x_init = *var_value;

            // Create modified value maps for derivative computation
            ParameterMap<real> values_plus = values;
            ParameterMap<real> values_minus = values;

            // Use 2-point central difference
            values_plus.insert_or_assign(variable_name.c_str(), x_init + h);
            const real y_plus = compound_evaluator(values_plus);

            values_minus.insert_or_assign(variable_name.c_str(), x_init - h);
            const real y_minus = compound_evaluator(values_minus);

            real derivative = (y_plus - y_minus) / (real(2) * h);

            // Check for numerical issues and use adaptive step if needed
            if (std::abs(derivative) > real(1e6)) {
                real smaller_h = h * real(0.1);
                values_plus.insert_or_assign(
                    variable_name.c_str(), x_init + smaller_h);
                const real y_plus_small = compound_evaluator(values_plus);

                values_minus.insert_or_assign(
                    variable_name.c_str(), x_init - smaller_h);
                const real y_minus_small = compound_evaluator(values_minus);

                derivative =
                    (y_plus_small - y_minus_small) / (real(2) * smaller_h);
            }

            return derivative;
        };
    }

    Expression::Expression(const std::string& expr_str)
        : expression_string_(expr_str),
          is_compound_(false)
    {
    }
    Expression::Expression(
        const std::string& expr_str,
        const std::vector<std::string>& variable_names)
        : expression_string_(expr_str),
          variable_names_(variable_names),
          is_compound_(false)
    {
    }
    Expression::Expression(
        const Expression& outer_expr,
        const std::vector<std::pair<const char*, Expression>>&
            variable_substitutions)
        : outer_expression_(std::make_unique<Expression>(outer_expr)),
          substitution_map_(variable_substitutions),
          is_compound_(true)
    {
        // Build compound expression string for display
        expression_string_ = outer_expr.get_string();
        for (const auto& pair : variable_substitutions) {
            expression_string_ += " with " + std::string(pair.first) + "=(" +
                                  pair.second.get_string() + ")";
        }
    }
    Expression::Expression(
        const Expression& outer_expr,
        std::initializer_list<std::pair<const char*, Expression>> substitutions)
        : outer_expression_(std::make_unique<Expression>(outer_expr)),
          substitution_map_(substitutions.begin(), substitutions.end()),
          is_compound_(true)
    {
        // Build compound expression string for display
        expression_string_ = outer_expr.get_string();
        for (const auto& pair : substitution_map_) {
            expression_string_ += " with " + std::string(pair.first) + "=(" +
                                  pair.second.get_string() + ")";
        }
    }
    Expression::Expression(const Expression& other)
        : expression_string_(other.expression_string_),
          variable_names_(other.variable_names_),
          is_parsed_(false),  // Force re-parsing with new symbol table
          is_compound_(other.is_compound_),
          outer_expression_(
              other.outer_expression_
                  ? std::make_unique<Expression>(*other.outer_expression_)
                  : nullptr),
          substitution_map_(other.substitution_map_),
          derivative_evaluator_(other.derivative_evaluator_),
          bound_variables_(other.bound_variables_)
    {
    }
    Expression& Expression::operator=(const Expression& other)
    {
        if (this != &other) {
            expression_string_ = other.expression_string_;
            variable_names_ = other.variable_names_;
            is_parsed_ = false;  // Force re-parsing
            compiled_expression_.reset();
            symbol_table_.reset();
            is_compound_ = other.is_compound_;
            outer_expression_ =
                other.outer_expression_
                    ? std::make_unique<Expression>(*other.outer_expression_)
                    : nullptr;
            substitution_map_ = other.substitution_map_;
            derivative_evaluator_ = other.derivative_evaluator_;
            bound_variables_ = other.bound_variables_;
        }
        return *this;
    }
    Expression Expression::bind_variables(
        const ParameterMap<real>& bound_values) const
    {
        Expression closure = *this;

        // Merge bound values with existing ones
        for (std::size_t i = 0; i < bound_values.size(); ++i) {
            const char* name = bound_values.get_name_at(i);
            const real& value = bound_values.get_value_at(i);
            closure.bound_variables_.insert_or_assign(name, value);
        }

        return closure;
    }

    Expression Expression::bind_variable(
        const std::string& var_name,
        real value) const
    {
        Expression closure = *this;
        closure.bound_variables_.insert_or_assign(var_name.c_str(), value);
        return closure;
    }

    real Expression::evaluate_at(
        const ParameterMap<real>& variable_values) const
    {
        // Handle expressions created from DerivativeExpression
        if (derivative_evaluator_) {
            // Merge bound variables with provided values
            ParameterMap<real> merged_values = bound_variables_;
            for (std::size_t i = 0; i < variable_values.size(); ++i) {
                const char* name = variable_values.get_name_at(i);
                const real& value = variable_values.get_value_at(i);
                merged_values.insert_or_assign(name, value);
            }
            return derivative_evaluator_(merged_values);
        }

        // Handle compound expressions
        if (is_compound_ && outer_expression_) {
            // Merge bound variables with provided values for substitution
            // evaluation
            ParameterMap<real> merged_values = bound_variables_;
            for (std::size_t i = 0; i < variable_values.size(); ++i) {
                const char* name = variable_values.get_name_at(i);
                const real& value = variable_values.get_value_at(i);
                merged_values.insert_or_assign(name, value);
            }

            // Evaluate substitutions
            ParameterMap<real> outer_values;
            for (std::size_t i = 0; i < substitution_map_.size(); ++i) {
                const auto& pair = substitution_map_[i];
                real sub_result = pair.second.evaluate_at(merged_values);
                outer_values.insert_or_assign(pair.first, sub_result);
            }

            return outer_expression_->evaluate_at(outer_values);
        }

        // Standard evaluation for non-compound expressions
        if (!is_parsed_ || !compiled_expression_) {
            // If no variables specified, discover them from the values provided
            if (variable_names_.empty()) {
                for (std::size_t i = 0; i < variable_values.size(); ++i) {
                    variable_names_.push_back(variable_values.get_name_at(i));
                }
                // Also add bound variable names
                for (std::size_t i = 0; i < bound_variables_.size(); ++i) {
                    variable_names_.push_back(bound_variables_.get_name_at(i));
                }
            }
            parse_expression();
        }

        if (!compiled_expression_) {
            throw std::runtime_error(
                "Expression not properly parsed: " + expression_string_);
        }

        // Merge bound variables with provided values (provided values take
        // precedence)
        ParameterMap<real> merged_values = bound_variables_;
        for (std::size_t i = 0; i < variable_values.size(); ++i) {
            const char* name = variable_values.get_name_at(i);
            const real& value = variable_values.get_value_at(i);
            merged_values.insert_or_assign(name, value);
        }

        // Set all variables in temp_variables_
        for (std::size_t i = 0; i < merged_values.size(); ++i) {
            const char* name = merged_values.get_name_at(i);
            const real& value = merged_values.get_value_at(i);
            auto ptr = temp_variables_.find(name);
            if (ptr)
                *ptr = value;
        }

        real result = compiled_expression_->value();
        return result;
    }

    Expression Expression::operator+(const Expression& other) const
    {
        Expression add_expr("x + y", { "x", "y" });
        return Expression(add_expr, { { "x", *this }, { "y", other } });
    }

    Expression Expression::operator-(const Expression& other) const
    {
        Expression sub_expr("x - y", { "x", "y" });
        return Expression(sub_expr, { { "x", *this }, { "y", other } });
    }

    Expression Expression::operator*(const Expression& other) const
    {
        Expression mul_expr("x * y", { "x", "y" });
        return Expression(mul_expr, { { "x", *this }, { "y", other } });
    }

    Expression Expression::operator/(const Expression& other) const
    {
        Expression div_expr("x / y", { "x", "y" });
        return Expression(div_expr, { { "x", *this }, { "y", other } });
    }

    Expression Expression::operator*(real scalar) const
    {
        Expression scalar_expr(std::to_string(scalar));
        Expression mul_expr("x * y", { "x", "y" });
        return Expression(mul_expr, { { "x", scalar_expr }, { "y", *this } });
    }

    Expression Expression::operator-() const
    {
        Expression neg_expr("-x", { "x" });
        return Expression(neg_expr, { { "x", *this } });
    }

    DerivativeExpression Expression::derivative(
        const std::string& variable_name) const
    {
        // Use more conservative step sizes for float precision
        real h;
        if (is_compound_ && outer_expression_) {
            // Check if any of the substitutions are derivatives
            bool has_derivative_substitution = false;
            for (const auto& pair : substitution_map_) {
                if (pair.second.derivative_evaluator_) {
                    has_derivative_substitution = true;
                    break;
                }
            }
            // For compound expressions with derivatives, use significantly
            // larger step
            h = has_derivative_substitution ? real(5e-3) : real(1e-4);
        }
        else if (derivative_evaluator_) {
            // This is already a derivative, so we're computing second
            // derivative
            h = real(5e-3);
        }
        else {
            // Simple expression
            h = real(1e-4);
        }

        // For compound expressions, use numerical chain rule
        if (is_compound_ && outer_expression_) {
            auto compound_evaluator = [this](const ParameterMap<real>& values) {
                return this->evaluate_at(values);
            };
            auto derivative_func = create_compound_derivative_function(
                compound_evaluator, variable_name, h);
            return DerivativeExpression(derivative_func, variable_name);
        }
        // Handle derivative expressions (derivatives of derivatives)
        else if (derivative_evaluator_) {
            auto derivative_func = create_compound_derivative_function(
                derivative_evaluator_, variable_name, h);
            return DerivativeExpression(derivative_func, variable_name);
        }
        else {
            // For simple expressions, use string-based derivative
            auto derivative_func = create_derivative_function(
                expression_string_, variable_name, h);
            return DerivativeExpression(derivative_func, variable_name);
        }
    }
    void Expression::parse_expression() const
    {
        symbol_table_ = std::make_unique<symbol_table_type>();
        compiled_expression_ = std::make_unique<expression_type>();

        // Enable constants and standard functions
        symbol_table_->add_constants();

        // Register specified variables only
        if (!variable_names_.empty()) {
            for (const auto& var_name : variable_names_) {
                auto* temp_var_ptr = temp_variables_.find(var_name.c_str());
                if (!temp_var_ptr) {
                    temp_variables_.insert_or_assign(var_name.c_str(), real(0));
                    temp_var_ptr = temp_variables_.find(var_name.c_str());
                }
                symbol_table_->add_variable(
                    var_name, const_cast<real&>(*temp_var_ptr));
            }
        }

        // Register symbol table with expression
        compiled_expression_->register_symbol_table(*symbol_table_);

        parser_type parser;
        if (!parser.compile(expression_string_, *compiled_expression_)) {
            throw std::runtime_error(
                "Failed to parse expression: " + expression_string_);
        }

        is_parsed_ = true;
    }
    Expression operator*(real scalar, const Expression& expr)
    {
        return expr * scalar;
    }
    Expression make_expression(const std::string& expr_str)
    {
        return Expression::from_string(expr_str);
    }

    std::vector<DerivativeExpression> Expression::gradient(
        const std::vector<std::string>& variable_names) const
    {
        std::vector<DerivativeExpression> grad;
        grad.reserve(variable_names.size());
        for (const auto& var : variable_names) {
            grad.push_back(derivative(var));
        }
        return grad;
    }

    // Create mapping expressions for coordinate transformation
    Expression create_mapping_expression(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec2d>& world_vertices)
    {
        if (world_vertices.size() != barycentric_names.size() + 1) {
            throw std::invalid_argument(
                "Invalid number of vertices for mapping");
        }

        // Create linear interpolation expressions for x and y
        std::string x_expr = std::to_string(world_vertices[0][0]);
        std::string y_expr = std::to_string(world_vertices[0][1]);

        for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
            real dx = world_vertices[i + 1][0] - world_vertices[0][0];
            real dy = world_vertices[i + 1][1] - world_vertices[0][1];

            if (dx != 0) {
                x_expr +=
                    " + " + std::to_string(dx) + " * " + barycentric_names[i];
            }
            if (dy != 0) {
                y_expr +=
                    " + " + std::to_string(dy) + " * " + barycentric_names[i];
            }
        }

        // Create a compound expression that sets x and y
        Expression x_mapping(x_expr, barycentric_names);
        Expression y_mapping(y_expr, barycentric_names);

        // Return a special mapping expression (this might need custom handling)
        return x_mapping;  // Simplified for now - might need special mapping
                           // expression type
    }

    Expression create_mapping_expression(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec3d>& world_vertices)
    {
        if (world_vertices.size() != barycentric_names.size() + 1) {
            throw std::invalid_argument(
                "Invalid number of vertices for mapping");
        }

        // Create linear interpolation expressions for x, y, and z
        std::string x_expr = std::to_string(world_vertices[0][0]);
        std::string y_expr = std::to_string(world_vertices[0][1]);
        std::string z_expr = std::to_string(world_vertices[0][2]);

        for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
            real dx = world_vertices[i + 1][0] - world_vertices[0][0];
            real dy = world_vertices[i + 1][1] - world_vertices[0][1];
            real dz = world_vertices[i + 1][2] - world_vertices[0][2];

            if (dx != 0) {
                x_expr +=
                    " + " + std::to_string(dx) + " * " + barycentric_names[i];
            }
            if (dy != 0) {
                y_expr +=
                    " + " + std::to_string(dy) + " * " + barycentric_names[i];
            }
            if (dz != 0) {
                z_expr +=
                    " + " + std::to_string(dz) + " * " + barycentric_names[i];
            }
        }

        // Create compound expression that sets x, y, z
        Expression x_mapping(x_expr, barycentric_names);
        Expression y_mapping(y_expr, barycentric_names);
        Expression z_mapping(z_expr, barycentric_names);

        return x_mapping;  // Simplified for now
    }

    // Create coordinate mapping expressions that bind world vertex coordinates
    ParameterMap<Expression> create_coordinate_mapping(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec2d>& world_vertices)
    {
        if (world_vertices.size() != barycentric_names.size() + 1) {
            throw std::invalid_argument(
                "Invalid number of vertices for mapping");
        }

        ParameterMap<Expression> coord_mapping;

        // Create linear interpolation expressions for x and y coordinates
        std::string x_expr = std::to_string(world_vertices[0][0]);
        std::string y_expr = std::to_string(world_vertices[0][1]);

        for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
            real dx = world_vertices[i + 1][0] - world_vertices[0][0];
            real dy = world_vertices[i + 1][1] - world_vertices[0][1];

            if (dx != 0) {
                x_expr +=
                    " + " + std::to_string(dx) + " * " + barycentric_names[i];
            }
            if (dy != 0) {
                y_expr +=
                    " + " + std::to_string(dy) + " * " + barycentric_names[i];
            }
        }

        // Create expressions for each coordinate
        Expression x_mapping(x_expr, barycentric_names);
        Expression y_mapping(y_expr, barycentric_names);

        // Store in coordinate mapping
        coord_mapping.insert_or_assign("x", x_mapping);
        coord_mapping.insert_or_assign("y", y_mapping);

        return coord_mapping;
    }

    ParameterMap<Expression> create_coordinate_mapping(
        const std::vector<std::string>& barycentric_names,
        const std::vector<pxr::GfVec3d>& world_vertices)
    {
        if (world_vertices.size() != barycentric_names.size() + 1) {
            throw std::invalid_argument(
                "Invalid number of vertices for mapping");
        }

        ParameterMap<Expression> coord_mapping;

        // Create linear interpolation expressions for x, y, and z coordinates
        std::string x_expr = std::to_string(world_vertices[0][0]);
        std::string y_expr = std::to_string(world_vertices[0][1]);
        std::string z_expr = std::to_string(world_vertices[0][2]);

        for (std::size_t i = 0; i < barycentric_names.size(); ++i) {
            real dx = world_vertices[i + 1][0] - world_vertices[0][0];
            real dy = world_vertices[i + 1][1] - world_vertices[0][1];
            real dz = world_vertices[i + 1][2] - world_vertices[0][2];

            if (dx != 0) {
                x_expr +=
                    " + " + std::to_string(dx) + " * " + barycentric_names[i];
            }
            if (dy != 0) {
                y_expr +=
                    " + " + std::to_string(dy) + " * " + barycentric_names[i];
            }
            if (dz != 0) {
                z_expr +=
                    " + " + std::to_string(dz) + " * " + barycentric_names[i];
            }
        }

        // Create expressions for each coordinate
        Expression x_mapping(x_expr, barycentric_names);
        Expression y_mapping(y_expr, barycentric_names);
        Expression z_mapping(z_expr, barycentric_names);

        // Store in coordinate mapping
        coord_mapping.insert_or_assign("x", x_mapping);
        coord_mapping.insert_or_assign("y", y_mapping);
        coord_mapping.insert_or_assign("z", z_mapping);

        return coord_mapping;
    }

    // Helper to create mapped expression using coordinate mapping
    Expression create_mapped_expression_with_coord_mapping(
        const Expression& expr,
        const ParameterMap<Expression>& coord_mapping,
        const std::vector<std::string>& barycentric_names)
    {
        // Create substitution map from coordinate mapping
        std::vector<std::pair<const char*, Expression>> substitutions;

        for (std::size_t i = 0; i < coord_mapping.size(); ++i) {
            const char* coord_name = coord_mapping.get_name_at(i);
            const Expression& coord_expr = coord_mapping.get_value_at(i);
            substitutions.emplace_back(coord_name, coord_expr);
        }

        // Create compound expression that substitutes physical coordinates
        return Expression(expr, substitutions);
    }

    // Create template function that is missing
    template<typename MappingExpr>
    Expression create_mapped_expression(
        const Expression& expr,
        const MappingExpr& mapping_expr,
        const std::vector<std::string>& barycentric_names)
    {
        // For now, return the original expression as fallback
        // This would need specific implementation based on MappingExpr type
        return expr;
    }

    // Explicit instantiation for ParameterMap<Expression>
    template Expression create_mapped_expression<ParameterMap<Expression>>(
        const Expression& expr,
        const ParameterMap<Expression>& mapping_expr,
        const std::vector<std::string>& barycentric_names);

}  // namespace fem_bem
}  // namespace USTC_CG
