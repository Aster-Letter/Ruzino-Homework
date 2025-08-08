#pragma once
#include <exprtk/exprtk.hpp>
#include <limits>

namespace USTC_CG {
namespace fem_bem {

    // Multi-variable Simpson integration on simplex using barycentric
    // coordinates
    template<typename T>
    inline T integrate_simplex(
        const exprtk::expression<T>& e,
        const std::vector<std::string>& barycentric_names,
        const std::size_t number_of_intervals = 100)
    {
        if (barycentric_names.empty())
            return T(0);

        const std::size_t dim = barycentric_names.size();
        const exprtk::symbol_table<T>& sym_table = e.get_symbol_table();

        if (!sym_table.valid())
            return std::numeric_limits<T>::quiet_NaN();

        // Get variable references
        std::vector<exprtk::details::variable_node<T>*> vars(dim);
        std::vector<T> original_values(dim);

        for (std::size_t i = 0; i < dim; ++i) {
            vars[i] = sym_table.get_variable(barycentric_names[i]);
            if (!vars[i])
                return std::numeric_limits<T>::quiet_NaN();
            original_values[i] = vars[i]->ref();
        }

        // Create evaluator that sets variables and evaluates expression
        auto evaluator = [&](const std::vector<T>& coords) -> T {
            for (std::size_t i = 0; i < dim; ++i) {
                vars[i]->ref() = coords[i];
            }
            return e.value();
        };

        T result = integrate_simplex_generic<T>(
            evaluator, barycentric_names, number_of_intervals);

        // Restore original values
        for (std::size_t i = 0; i < dim; ++i) {
            vars[i]->ref() = original_values[i];
        }

        return result;
    }

    // Integration of product of two expressions
    template<typename T>
    inline T integrate_product_simplex(
        const exprtk::expression<T>& expr1,
        const exprtk::expression<T>& expr2,
        const std::vector<std::string>& barycentric_names,
        const std::size_t number_of_intervals = 100)
    {
        if (barycentric_names.empty())
            return T(0);

        const std::size_t dim = barycentric_names.size();
        const exprtk::symbol_table<T>& sym_table1 = expr1.get_symbol_table();
        const exprtk::symbol_table<T>& sym_table2 = expr2.get_symbol_table();

        if (!sym_table1.valid() || !sym_table2.valid())
            return std::numeric_limits<T>::quiet_NaN();

        // Get variable references for both expressions
        std::vector<exprtk::details::variable_node<T>*> vars1(dim);
        std::vector<exprtk::details::variable_node<T>*> vars2(dim);
        std::vector<T> original_values1(dim), original_values2(dim);

        for (std::size_t i = 0; i < dim; ++i) {
            vars1[i] = sym_table1.get_variable(barycentric_names[i]);
            vars2[i] = sym_table2.get_variable(barycentric_names[i]);
            if (!vars1[i] || !vars2[i])
                return std::numeric_limits<T>::quiet_NaN();
            original_values1[i] = vars1[i]->ref();
            original_values2[i] = vars2[i]->ref();
        }

        // Create evaluator for product
        auto evaluator = [&](const std::vector<T>& coords) -> T {
            for (std::size_t i = 0; i < dim; ++i) {
                vars1[i]->ref() = coords[i];
                vars2[i]->ref() = coords[i];
            }
            return expr1.value() * expr2.value();
        };

        T result = integrate_simplex_generic<T>(
            evaluator, barycentric_names, number_of_intervals);

        // Restore original values
        for (std::size_t i = 0; i < dim; ++i) {
            vars1[i]->ref() = original_values1[i];
            vars2[i]->ref() = original_values2[i];
        }

        return result;
    }

    // Generic simplex integration framework
    template<typename T, typename EvaluatorFunc>
    inline T integrate_simplex_generic(
        EvaluatorFunc evaluator,
        const std::vector<std::string>& barycentric_names,
        std::size_t number_of_intervals = 100)
    {
        if (barycentric_names.empty())
            return T(0);

        const std::size_t dim = barycentric_names.size();
        T total_integral = T(0);

        // For 1D case (line segment)
        if (dim == 1) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i < number_of_intervals; ++i) {
                const T u1 = i * h;
                const T u2 = (i + 1) * h;
                const T u_mid = (u1 + u2) / T(2);

                // Simpson's rule over [u1, u2]
                const T y1 = evaluator(std::vector<T>{ u1 });
                const T y_mid = evaluator(std::vector<T>{ u_mid });
                const T y2 = evaluator(std::vector<T>{ u2 });

                total_integral += h * (y1 + T(4) * y_mid + y2) / T(6);
            }
        }
        // For 2D case (triangle)
        else if (dim == 2) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i <= number_of_intervals; ++i) {
                for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                    const T u1 = i * h;
                    const T u2 = j * h;

                    if (u1 + u2 <= T(1)) {
                        T weight = T(1);
                        // Boundary correction for trapezoidal rule
                        if (i == 0 || j == 0 || i + j == number_of_intervals)
                            weight = T(0.5);
                        if ((i == 0 && j == 0) ||
                            (i == 0 && i + j == number_of_intervals) ||
                            (j == 0 && i + j == number_of_intervals))
                            weight = T(0.25);

                        total_integral += weight *
                                          evaluator(std::vector<T>{ u1, u2 }) *
                                          h * h * T(2);
                    }
                }
            }
        }
        // For 3D case (tetrahedron)
        else if (dim == 3) {
            const T h = T(1) / number_of_intervals;

            for (std::size_t i = 0; i <= number_of_intervals; ++i) {
                for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                    for (std::size_t k = 0; k <= number_of_intervals - i - j;
                         ++k) {
                        const T u1 = i * h;
                        const T u2 = j * h;
                        const T u3 = k * h;

                        if (u1 + u2 + u3 <= T(1)) {
                            T weight = T(1);
                            // Boundary correction
                            int boundary_count = 0;
                            if (i == 0)
                                boundary_count++;
                            if (j == 0)
                                boundary_count++;
                            if (k == 0)
                                boundary_count++;
                            if (i + j + k == number_of_intervals)
                                boundary_count++;

                            if (boundary_count > 0)
                                weight = T(1) / T(1 << boundary_count);

                            total_integral +=
                                weight *
                                evaluator(std::vector<T>{ u1, u2, u3 }) * h *
                                h * h * T(6);
                        }
                    }
                }
            }
        }

        return total_integral;
    }

    // Integration of product with mapping function support
    template<typename T, typename MappingFunc>
    inline T integrate_product_simplex_with_mapping(
        const exprtk::expression<T>& expr1,
        const exprtk::expression<T>& expr2,
        MappingFunc mapping,
        const std::vector<std::string>& barycentric_names,
        const std::size_t number_of_intervals = 100)
    {
        if (barycentric_names.empty())
            return T(0);

        const std::size_t dim = barycentric_names.size();
        const exprtk::symbol_table<T>& sym_table1 = expr1.get_symbol_table();
        const exprtk::symbol_table<T>& sym_table2 = expr2.get_symbol_table();

        if (!sym_table1.valid() || !sym_table2.valid())
            return std::numeric_limits<T>::quiet_NaN();

        // Get variable references for barycentric coordinates
        std::vector<exprtk::details::variable_node<T>*> barycentric_vars1(dim);
        std::vector<exprtk::details::variable_node<T>*> barycentric_vars2(dim);
        std::vector<T> original_values1(dim), original_values2(dim);

        for (std::size_t i = 0; i < dim; ++i) {
            barycentric_vars1[i] =
                sym_table1.get_variable(barycentric_names[i]);
            barycentric_vars2[i] =
                sym_table2.get_variable(barycentric_names[i]);
            if (!barycentric_vars1[i] || !barycentric_vars2[i])
                return std::numeric_limits<T>::quiet_NaN();
            original_values1[i] = barycentric_vars1[i]->ref();
            original_values2[i] = barycentric_vars2[i]->ref();
        }

        // Get physical coordinate variables in expr2 symbol table
        exprtk::details::variable_node<T>* x_var = sym_table2.get_variable("x");
        exprtk::details::variable_node<T>* y_var = sym_table2.get_variable("y");
        exprtk::details::variable_node<T>* z_var = sym_table2.get_variable("z");

        T x_orig = x_var ? x_var->ref() : T(0);
        T y_orig = y_var ? y_var->ref() : T(0);
        T z_orig = z_var ? z_var->ref() : T(0);

        // Create evaluator that handles coordinate mapping
        auto evaluator = [&](const std::vector<T>& coords) -> T {
            // Set barycentric coordinates
            for (std::size_t i = 0; i < dim; ++i) {
                barycentric_vars1[i]->ref() = coords[i];
                barycentric_vars2[i]->ref() = coords[i];
            }

            // Apply mapping to get physical coordinates
            if constexpr (std::is_invocable_v<MappingFunc, T>) {
                if (dim == 1) {
                    auto mapped_coords = mapping(coords[0]);
                    if (x_var && mapped_coords.size() > 0)
                        x_var->ref() = mapped_coords[0];
                    if (y_var && mapped_coords.size() > 1)
                        y_var->ref() = mapped_coords[1];
                    if (z_var && mapped_coords.size() > 2)
                        z_var->ref() = mapped_coords[2];
                }
            }
            if constexpr (std::is_invocable_v<MappingFunc, T, T>) {
                if (dim == 2) {
                    auto mapped_coords = mapping(coords[0], coords[1]);
                    if (x_var && mapped_coords.size() > 0)
                        x_var->ref() = mapped_coords[0];
                    if (y_var && mapped_coords.size() > 1)
                        y_var->ref() = mapped_coords[1];
                    if (z_var && mapped_coords.size() > 2)
                        z_var->ref() = mapped_coords[2];
                }
            }
            if constexpr (std::is_invocable_v<MappingFunc, T, T, T>) {
                if (dim == 3) {
                    auto mapped_coords =
                        mapping(coords[0], coords[1], coords[2]);
                    if (x_var && mapped_coords.size() > 0)
                        x_var->ref() = mapped_coords[0];
                    if (y_var && mapped_coords.size() > 1)
                        y_var->ref() = mapped_coords[1];
                    if (z_var && mapped_coords.size() > 2)
                        z_var->ref() = mapped_coords[2];
                }
            }

            return expr1.value() * expr2.value();
        };

        T result = integrate_simplex_generic<T>(
            evaluator, barycentric_names, number_of_intervals);

        // Restore original values
        for (std::size_t i = 0; i < dim; ++i) {
            barycentric_vars1[i]->ref() = original_values1[i];
            barycentric_vars2[i]->ref() = original_values2[i];
        }
        if (x_var)
            x_var->ref() = x_orig;
        if (y_var)
            y_var->ref() = y_orig;
        if (z_var)
            z_var->ref() = z_orig;

        return result;
    }

}  // namespace fem_bem

}  // namespace USTC_CG
