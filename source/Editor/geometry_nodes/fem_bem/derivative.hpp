#pragma once
#include <exprtk/exprtk.hpp>
#include <functional>
#include <string>
#include <unordered_map>

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
            std::function<T(const std::unordered_map<std::string, T>&)> func,
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
        const T& h = T(0.00000001))
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
    std::function<T(const std::unordered_map<std::string, T>&)>
    create_derivative_function(
        const std::string& expression_string,
        const std::string& variable_name,
        const T& h = T(0.00000001))
    {
        return [expression_string, variable_name, h](
                   const std::unordered_map<std::string, T>& values) -> T {
            // Create a temporary expression for derivative computation
            exprtk::symbol_table<T> symbol_table;
            exprtk::expression<T> expr;
            exprtk::parser<T> parser;

            // Add all variables to symbol table
            std::unordered_map<std::string, T> temp_values = values;

            for (auto& [name, value] : temp_values) {
                symbol_table.add_variable(name, value);
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
    std::function<T(const std::unordered_map<std::string, T>&)>
    create_compound_derivative_function(
        const std::function<T(const std::unordered_map<std::string, T>&)>&
            compound_evaluator,
        const std::string& variable_name,
        const T& h = T(0.00000001))
    {
        return [compound_evaluator, variable_name, h](
                   const std::unordered_map<std::string, T>& values) -> T {
            auto values_iter = values.find(variable_name);
            if (values_iter == values.end()) {
                return T(0);  // Variable not found
            }

            const T x_init = values_iter->second;
            const T _2h = T(2) * h;

            // Create modified value maps for derivative computation
            std::unordered_map<std::string, T> values_plus_2h = values;
            std::unordered_map<std::string, T> values_plus_h = values;
            std::unordered_map<std::string, T> values_minus_h = values;
            std::unordered_map<std::string, T> values_minus_2h = values;

            values_plus_2h[variable_name] = x_init + _2h;
            values_plus_h[variable_name] = x_init + h;
            values_minus_h[variable_name] = x_init - h;
            values_minus_2h[variable_name] = x_init - _2h;

            // 5-point stencil numerical derivative
            const T y0 = compound_evaluator(values_plus_2h);
            const T y1 = compound_evaluator(values_plus_h);
            const T y2 = compound_evaluator(values_minus_h);
            const T y3 = compound_evaluator(values_minus_2h);

            return (-y0 + T(8) * (y1 - y2) + y3) / (T(12) * h);
        };
    }

}  // namespace fem_bem
}  // namespace USTC_CG