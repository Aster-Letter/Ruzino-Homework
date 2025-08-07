#pragma once
#include <exprtk/exprtk.hpp>
#include <stdexcept>

#include "pxr/base/gf/vec3f.h"
USTC_CG_NAMESPACE_OPEN_SCOPE

namespace fem_bem {

// Multi-variable Simpson integration on simplex using barycentric coordinates
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

    T total_integral = T(0);

    // For 2D case (triangle)
    if (dim == 3) {
        const T h = T(1) / number_of_intervals;

        for (std::size_t i = 0; i <= number_of_intervals; ++i) {
            for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                const T u1 = i * h;
                const T u2 = j * h;
                const T u3 = T(1) - u1 - u2;

                if (u3 >= T(0)) {
                    vars[0]->ref() = u1;
                    vars[1]->ref() = u2;
                    vars[2]->ref() = u3;

                    T weight = T(1);
                    // Boundary correction for trapezoidal rule
                    if (i == 0 || j == 0 || i + j == number_of_intervals)
                        weight = T(0.5);
                    if ((i == 0 && j == 0) ||
                        (i == 0 && i + j == number_of_intervals) ||
                        (j == 0 && i + j == number_of_intervals))
                        weight = T(0.25);

                    // Correct area element for triangle: h*h/2 * 2 = h*h
                    total_integral += weight * e.value() * h * h * T(2);
                }
            }
        }
    }
    // For 1D case (line segment)
    else if (dim == 2) {
        const T h = T(1) / (T(2) * number_of_intervals);

        for (std::size_t i = 0; i < number_of_intervals; ++i) {
            T u = i * T(2) * h;

            // Simpson's rule: f(x0) + 4f(x1) + f(x2)
            vars[0]->ref() = u;
            vars[1]->ref() = T(1) - u;
            const T y0 = e.value();

            u += h;
            vars[0]->ref() = u;
            vars[1]->ref() = T(1) - u;
            const T y1 = e.value();

            u += h;
            vars[0]->ref() = u;
            vars[1]->ref() = T(1) - u;
            const T y2 = e.value();

            total_integral += h * (y0 + T(4) * y1 + y2) / T(3);
        }
    }
    // For 3D case (tetrahedron)
    else if (dim == 4) {
        const T h = T(1) / number_of_intervals;

        for (std::size_t i = 0; i <= number_of_intervals; ++i) {
            for (std::size_t j = 0; j <= number_of_intervals - i; ++j) {
                for (std::size_t k = 0; k <= number_of_intervals - i - j; ++k) {
                    const T u1 = i * h;
                    const T u2 = j * h;
                    const T u3 = k * h;
                    const T u4 = T(1) - u1 - u2 - u3;

                    if (u4 >= T(0)) {
                        vars[0]->ref() = u1;
                        vars[1]->ref() = u2;
                        vars[2]->ref() = u3;
                        vars[3]->ref() = u4;

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

                        total_integral += weight * e.value() * h * h * h * T(6);
                    }
                }
            }
        }
    }

    // Restore original values
    for (std::size_t i = 0; i < dim; ++i) {
        vars[i]->ref() = original_values[i];
    }

    return total_integral;
}

}  // namespace fem_bem

USTC_CG_NAMESPACE_CLOSE_SCOPE
