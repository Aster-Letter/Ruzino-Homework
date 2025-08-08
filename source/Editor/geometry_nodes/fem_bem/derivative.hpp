#pragma once
#include <exprtk/exprtk.hpp>
#include <functional>
#include <string>

#include "parameter_map.hpp"

namespace USTC_CG {
namespace fem_bem {

    // Forward declaration
    template<typename T>
    class Expression;

    // Numerical derivative class that inherits from Expression
    template<typename T>
    class DerivativeExpression : public Expression<T> {
       private:
        std::string variable_name_;

       public:
        DerivativeExpression(
            std::function<T(const ParameterMap<T>&)> func,
            const std::string& var_name)
            : Expression<T>(),
              variable_name_(var_name)
        {
            // Set up the base class with the derivative evaluator
            this->derivative_evaluator_ = std::move(func);
            this->expression_string_ =
                "";  // Derivatives don't have string representation
            this->is_compound_ = false;
        }

        // Get the variable name for this derivative
        const std::string& get_variable_name() const
        {
            return variable_name_;
        }

        // Override is_string_based to return false for derivatives
        bool is_string_based() const override
        {
            return false;
        }

        // Note: Integration methods are inherited from Expression base class
    };

    // Numerical derivative using finite differences (5-point stencil)
    template<typename T>
    inline T numerical_derivative(
        const exprtk::expression<T>& expr,
        exprtk::details::variable_node<T>* var,
        const T& h = T(1e-8))
    {
        const T x_init = var->ref();
        const T _2h = T(2) * h;

        var->ref() = x_init + _2h;
        const T y0 = expr.value();
        var->ref() = x_init + h;
        const T y1 = expr.value();
        var->ref() = x_init - h;
        const T y2 = expr.value();
        var->ref() = x_init - _2h;
        const T y3 = expr.value();
        var->ref() = x_init;

        return (-y0 + T(8) * (y1 - y2) + y3) / (T(12) * h);
    }

    // Create a derivative function for an expression
    template<typename T>
    std::function<T(const ParameterMap<T>&)> create_derivative_function(
        const std::string& expression_string,
        const std::string& variable_name,
        const T& h = T(1e-8))
    {
        return [expression_string, variable_name, h](
                   const ParameterMap<T>& values) -> T {
            // Create a temporary expression for derivative computation
            exprtk::symbol_table<T> symbol_table;
            exprtk::expression<T> expr;
            exprtk::parser<T> parser;

            // Add all variables to symbol table
            ParameterMap<T> temp_values = values;

            for (const auto& pair : temp_values) {
                symbol_table.add_variable(
                    pair.first, const_cast<T&>(pair.second));
            }

            symbol_table.add_constants();
            expr.register_symbol_table(symbol_table);

            if (!parser.compile(expression_string, expr)) {
                return T(0);  // Return 0 for invalid expressions
            }

            // Find the variable node for differentiation
            auto* var_node = symbol_table.get_variable(variable_name);
            if (!var_node) {
                return T(0);  // Variable not found
            }

            return numerical_derivative(expr, var_node, h);
        };
    }

    // Create a derivative function for compound expressions using chain rule
    template<typename T>
    std::function<T(const ParameterMap<T>&)>
    create_compound_derivative_function(
        const std::function<T(const ParameterMap<T>&)>& compound_evaluator,
        const std::string& variable_name,
        const T& h = T(1e-6))
    {
        return [compound_evaluator, variable_name, h](
                   const ParameterMap<T>& values) -> T {
            const T* var_value = values.find(variable_name.c_str());
            if (!var_value) {
                return T(0);  // Variable not found
            }

            const T x_init = *var_value;

            // Create modified value maps for derivative computation
            ParameterMap<T> values_plus_h = values;
            ParameterMap<T> values_minus_h = values;

            values_plus_h.insert_or_assign(variable_name.c_str(), x_init + h);
            values_minus_h.insert_or_assign(variable_name.c_str(), x_init - h);

            // Use simple 2-point central difference for better numerical
            // stability
            const T y_plus = compound_evaluator(values_plus_h);
            const T y_minus = compound_evaluator(values_minus_h);

            return (y_plus - y_minus) / (T(2) * h);
        };
    }

}  // namespace fem_bem
}  // namespace USTC_CG
