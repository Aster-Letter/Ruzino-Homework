#pragma once
#include <exprtk/exprtk.hpp>
#include <memory>
#include <string>
#include <vector>

#include "derivative.hpp"
#include "integrate.hpp"
#include "parameter_map.hpp"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec3d.h"

namespace USTC_CG {
namespace fem_bem {

    // Forward declarations
    template<typename T>
    class DerivativeExpression;

    template<typename T = double>
    class Expression {
       public:
        using value_type = T;
        using expression_type = exprtk::expression<T>;
        using symbol_table_type = exprtk::symbol_table<T>;
        using parser_type = exprtk::parser<T>;

        // Constructors
        Expression() = default;
        explicit Expression(const std::string& expr_str)
            : expression_string_(expr_str),
              is_compound_(false)
        {
        }

        Expression(
            const std::string& expr_str,
            const std::vector<std::string>& variable_names)
            : expression_string_(expr_str),
              variable_names_(variable_names),
              is_compound_(false)
        {
        }

        // Compound expression constructor
        Expression(
            const Expression& outer_expr,
            const std::vector<std::pair<std::string, Expression>>&
                variable_substitutions)
            : outer_expression_(std::make_unique<Expression>(outer_expr)),
              substitution_map_(variable_substitutions),
              is_compound_(true)
        {
            // Build compound expression string for display
            expression_string_ = outer_expr.get_string();
            for (const auto& pair : variable_substitutions) {
                expression_string_ += " with " + pair.first + "=(" +
                                      pair.second.get_string() + ")";
            }
        }

        // Compound expression constructor with initializer list for convenience
        Expression(
            const Expression& outer_expr,
            std::initializer_list<std::pair<const char*, Expression>>
                substitutions)
            : outer_expression_(std::make_unique<Expression>(outer_expr)),
              substitution_map_(substitutions.begin(), substitutions.end()),
              is_compound_(true)
        {
            // Build compound expression string for display
            expression_string_ = outer_expr.get_string();
            for (const auto& pair : substitution_map_) {
                expression_string_ += " with " + std::string(pair.first) +
                                      "=(" + pair.second.get_string() + ")";
            }
        }

        // Copy constructor and assignment
        Expression(const Expression& other)
            : expression_string_(other.expression_string_),
              variable_names_(other.variable_names_),
              is_parsed_(false),  // Force re-parsing with new symbol table
              is_compound_(other.is_compound_),
              outer_expression_(
                  other.outer_expression_
                      ? std::make_unique<Expression>(*other.outer_expression_)
                      : nullptr),
              substitution_map_(other.substitution_map_),
              derivative_evaluator_(other.derivative_evaluator_)
        {
        }

        Expression& operator=(const Expression& other)
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
            }
            return *this;
        }

        // Move constructor and assignment
        Expression(Expression&& other) noexcept = default;
        Expression& operator=(Expression&& other) noexcept = default;

        // Virtual destructor for inheritance
        virtual ~Expression() = default;

        // Factory methods
        static Expression from_string(const std::string& expr_str)
        {
            return Expression(expr_str);
        }

        static Expression constant(T value)
        {
            return Expression(std::to_string(value));
        }

        static Expression zero()
        {
            return Expression("0");
        }

        static Expression one()
        {
            return Expression("1");
        }

        // Basic properties
        const std::string& get_string() const
        {
            return expression_string_;
        }

        // Method to check if this is a string-based expression
        virtual bool is_string_based() const
        {
            return true;  // Regular expressions are always string-based
        }

        // Evaluation
        T evaluate() const
        {
            ensure_parsed();
            if (!compiled_expression_) {
                throw std::runtime_error(
                    "Expression not properly parsed: " + expression_string_);
            }
            return compiled_expression_->value();
        }

        T evaluate_at(const ParameterMap<T>& variable_values) const
        {
            // Handle expressions created from DerivativeExpression
            if (derivative_evaluator_) {
                return derivative_evaluator_(variable_values);
            }

            // Handle compound expressions
            if (is_compound_ && outer_expression_) {
                // Evaluate substitutions first
                ParameterMap<T> outer_values;

                for (const auto& pair : substitution_map_) {
                    T sub_result = pair.second.evaluate_at(variable_values);
                    outer_values.insert_or_assign(pair.first, sub_result);
                }

                return outer_expression_->evaluate_at(outer_values);
            }

            // Standard evaluation for non-compound expressions
            if (!is_parsed_ || !compiled_expression_) {
                // If no variables specified, discover them from the values
                // provided
                if (variable_names_.empty()) {
                    for (const auto& pair : variable_values) {
                        variable_names_.push_back(pair.first);
                    }
                }
                parse_expression();
            }

            if (!compiled_expression_) {
                throw std::runtime_error(
                    "Expression not properly parsed: " + expression_string_);
            }

            // Store original values

            const exprtk::symbol_table<T>& sym_table =
                compiled_expression_->get_symbol_table();

            for (const auto& pair : variable_values) {
                *temp_variables_.find(pair.first) = pair.second;
                //auto* var_node = sym_table.get_variable_unchecked(pair.first);
                //if (var_node) {
                //    var_node->ref() = pair.second;
                //}
            }

            T result = compiled_expression_->value();

            return result;
        }

        // Arithmetic operations
        Expression operator+(const Expression& other) const
        {
            return Expression(
                "(" + expression_string_ + ") + (" + other.expression_string_ +
                ")");
        }

        Expression operator-(const Expression& other) const
        {
            return Expression(
                "(" + expression_string_ + ") - (" + other.expression_string_ +
                ")");
        }

        Expression operator*(const Expression& other) const
        {
            return Expression(
                "(" + expression_string_ + ") * (" + other.expression_string_ +
                ")");
        }

        Expression operator/(const Expression& other) const
        {
            return Expression(
                "(" + expression_string_ + ") / (" + other.expression_string_ +
                ")");
        }

        Expression operator*(T scalar) const
        {
            return Expression(
                std::to_string(scalar) + " * (" + expression_string_ + ")");
        }

        Expression operator-() const
        {
            return Expression("-(" + expression_string_ + ")");
        }

        // Integration methods
        template<typename MappingFunc = std::nullptr_t>
        T integrate_over_simplex(
            const std::vector<std::string>& barycentric_names,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            // Handle compound expressions using numerical integration
            if (derivative_evaluator_ || (is_compound_ && outer_expression_)) {
                auto evaluator = [this](const ParameterMap<T>& values) {
                    return this->evaluate_at(values);
                };
                return integrate_numerical_with_mapping(
                    evaluator, mapping, barycentric_names, intervals);
            }

            // For simple expressions, use existing integration methods
            ensure_parsed();
            if constexpr (std::is_same_v<MappingFunc, std::nullptr_t>) {
                return integrate_simplex(
                    *compiled_expression_, barycentric_names, intervals);
            }
            else {
                return integrate_simplex_with_mapping(
                    *compiled_expression_,
                    mapping,
                    barycentric_names,
                    intervals);
            }
        }

        template<typename MappingFunc = std::nullptr_t>
        T integrate_product_with(
            const Expression& other,
            const std::vector<std::string>& barycentric_names,
            MappingFunc mapping = nullptr,
            std::size_t intervals = 100) const
        {
            // If either expression is compound or derivative-based, use
            // numerical integration
            if ((derivative_evaluator_ ||
                 (is_compound_ && outer_expression_)) ||
                (other.derivative_evaluator_ ||
                 (other.is_compound_ && other.outer_expression_))) {
                auto product_evaluator =
                    [this, &other](const ParameterMap<T>& values) {
                        return this->evaluate_at(values) *
                               other.evaluate_at(values);
                    };
                return integrate_numerical_with_mapping(
                    product_evaluator, mapping, barycentric_names, intervals);
            }

            // For simple expressions, use existing integration methods
            ensure_parsed();
            other.ensure_parsed();

            if constexpr (std::is_same_v<MappingFunc, std::nullptr_t>) {
                return integrate_product_simplex(
                    *compiled_expression_,
                    *other.compiled_expression_,
                    barycentric_names,
                    intervals);
            }
            else {
                return integrate_product_simplex_with_mapping(
                    *compiled_expression_,
                    *other.compiled_expression_,
                    mapping,
                    barycentric_names,
                    intervals);
            }
        }

        // Derivative methods
        DerivativeExpression<T> derivative(
            const std::string& variable_name) const
        {
            // For compound expressions, use numerical chain rule
            if (is_compound_ && outer_expression_) {
                auto compound_evaluator =
                    [this](const ParameterMap<T>& values) {
                        return this->evaluate_at(values);
                    };
                auto derivative_func = create_compound_derivative_function<T>(
                    compound_evaluator, variable_name);
                return DerivativeExpression<T>(derivative_func, variable_name);
            }
            // Handle derivative expressions (derivatives of derivatives)
            else if (derivative_evaluator_) {
                auto derivative_func = create_compound_derivative_function<T>(
                    derivative_evaluator_, variable_name);
                return DerivativeExpression<T>(derivative_func, variable_name);
            }
            else {
                // For simple expressions, use string-based derivative
                auto derivative_func = create_derivative_function<T>(
                    expression_string_, variable_name);
                return DerivativeExpression<T>(derivative_func, variable_name);
            }
        }

        std::vector<DerivativeExpression<T>> gradient(
            const std::vector<std::string>& variable_names) const
        {
            std::vector<DerivativeExpression<T>> grad;
            grad.reserve(variable_names.size());
            for (const auto& var : variable_names) {
                grad.push_back(derivative(var));
            }
            return grad;
        }

        // Access to underlying exprtk objects (for advanced use)
        const expression_type* get_compiled_expression() const
        {
            ensure_parsed();
            return compiled_expression_.get();
        }

        const symbol_table_type* get_symbol_table() const
        {
            ensure_parsed();
            return symbol_table_.get();
        }

       protected:
        // Protected members for derived classes to access
        std::string expression_string_;
        mutable std::vector<std::string> variable_names_;

        // Parsed expression components
        mutable std::unique_ptr<symbol_table_type> symbol_table_;
        mutable std::unique_ptr<expression_type> compiled_expression_;
        mutable bool is_parsed_ = false;

        // Storage for variables
        mutable ParameterMap<T> temp_variables_;

        // Compound expression support
        bool is_compound_ = false;
        std::unique_ptr<Expression> outer_expression_;
        std::vector<std::pair<const char*, Expression>> substitution_map_;

        // Support for DerivativeExpression conversion
        std::function<T(const ParameterMap<T>&)> derivative_evaluator_;

        // Numerical integration for compound expressions
        template<typename EvaluatorFunc, typename MappingFunc = std::nullptr_t>
        T integrate_numerical_with_mapping(
            EvaluatorFunc evaluator,
            MappingFunc mapping,
            const std::vector<std::string>& barycentric_names,
            std::size_t intervals) const
        {
            // Create a unified evaluator that handles both coordinate types
            auto unified_evaluator = [&](const std::vector<T>& coords) -> T {
                // Convert vector coords to map format
                ParameterMap<T> values;

                // Set barycentric coordinates
                for (std::size_t i = 0;
                     i < coords.size() && i < barycentric_names.size();
                     ++i) {
                    values.insert_or_assign(
                        barycentric_names[i].c_str(), coords[i]);
                }

                // Apply mapping if provided
                if constexpr (!std::is_same_v<MappingFunc, std::nullptr_t>) {
                    if (coords.size() == 1) {
                        auto mapped_coords = mapping(coords[0]);
                        if (mapped_coords.size() > 0)
                            values.insert_or_assign("x", mapped_coords[0]);
                        if (mapped_coords.size() > 1)
                            values.insert_or_assign("y", mapped_coords[1]);
                        if (mapped_coords.size() > 2)
                            values.insert_or_assign("z", mapped_coords[2]);
                    }
                    else if (coords.size() == 2) {
                        auto mapped_coords = mapping(coords[0], coords[1]);
                        if (mapped_coords.size() > 0)
                            values.insert_or_assign("x", mapped_coords[0]);
                        if (mapped_coords.size() > 1)
                            values.insert_or_assign("y", mapped_coords[1]);
                        if (mapped_coords.size() > 2)
                            values.insert_or_assign("z", mapped_coords[2]);
                    }
                    else if (coords.size() == 3) {
                        auto mapped_coords =
                            mapping(coords[0], coords[1], coords[2]);
                        if (mapped_coords.size() > 0)
                            values.insert_or_assign("x", mapped_coords[0]);
                        if (mapped_coords.size() > 1)
                            values.insert_or_assign("y", mapped_coords[1]);
                        if (mapped_coords.size() > 2)
                            values.insert_or_assign("z", mapped_coords[2]);
                    }
                }

                return evaluator(values);
            };

            // Use the generic integration framework
            return integrate_simplex_generic<T>(
                unified_evaluator, barycentric_names, intervals);
        }

        void ensure_parsed() const
        {
            if (!is_parsed_ || !compiled_expression_) {
                parse_expression();
            }
        }

       private:
        void parse_expression() const
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
                        temp_variables_.insert_or_assign(
                            var_name.c_str(), T(0));
                        temp_var_ptr = temp_variables_.find(var_name.c_str());
                    }
                    symbol_table_->add_variable(
                        var_name, const_cast<T&>(*temp_var_ptr));
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

        template<typename MappingFunc>
        T integrate_simplex_with_mapping(
            const expression_type& expr,
            MappingFunc mapping,
            const std::vector<std::string>& barycentric_names,
            std::size_t intervals) const
        {
            // For compound expressions, use numerical integration
            if (is_compound_ && outer_expression_) {
                auto compound_evaluator =
                    [this](const ParameterMap<T>& values) {
                        return this->evaluate_at(values);
                    };
                return integrate_numerical_with_mapping(
                    compound_evaluator, mapping, barycentric_names, intervals);
            }

            // For simple expressions, use the mapping-aware integration from
            // integrate.hpp
            Expression<T> unit_expr("1", barycentric_names);
            unit_expr.ensure_parsed();

            return integrate_product_simplex_with_mapping(
                *unit_expr.compiled_expression_,
                expr,
                mapping,
                barycentric_names,
                intervals);
        }
    };

    // Type aliases
    using ExpressionD = Expression<double>;
    using ExpressionF = Expression<float>;

    // Free function operators for scalar * Expression
    template<typename T>
    Expression<T> operator*(T scalar, const Expression<T>& expr)
    {
        return expr * scalar;
    }

    // Utility functions
    template<typename T>
    Expression<T> make_expression(const std::string& expr_str)
    {
        return Expression<T>::from_string(expr_str);
    }

    inline ExpressionD make_expression_d(const std::string& expr_str)
    {
        return ExpressionD::from_string(expr_str);
    }

    inline ExpressionF make_expression_f(const std::string& expr_str)
    {
        return ExpressionF::from_string(expr_str);
    }

}  // namespace fem_bem
}  // namespace USTC_CG
