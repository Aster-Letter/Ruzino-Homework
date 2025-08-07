#include <exprtk/exprtk.hpp>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

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

// Test functions
void test_line_integral()
{
    typedef double T;
    typedef exprtk::symbol_table<T> symbol_table_t;
    typedef exprtk::expression<T> expression_t;
    typedef exprtk::parser<T> parser_t;

    T u1 = 0.0, u2 = 0.0;

    symbol_table_t symbol_table;
    symbol_table.add_variable("u1", u1);
    symbol_table.add_variable("u2", u2);

    expression_t expression;
    expression.register_symbol_table(symbol_table);

    parser_t parser;

    // Test 1: integrate u1 over line segment (should be 1/2)
    std::string expr1 = "u1";
    if (parser.compile(expr1, expression)) {
        std::vector<std::string> vars = { "u1", "u2" };
        T result = integrate_simplex(expression, vars, 1000);
        std::cout << "Line integral of u1: " << result << " (expected: 0.5)"
                  << std::endl;
    }

    // Test 2: integrate u1*u2 over line segment (should be 1/6)
    std::string expr2 = "u1*u2";
    if (parser.compile(expr2, expression)) {
        std::vector<std::string> vars = { "u1", "u2" };
        T result = integrate_simplex(expression, vars, 1000);
        std::cout << "Line integral of u1*u2: " << result
                  << " (expected: 0.1667)" << std::endl;
    }
}

void test_triangle_integral()
{
    typedef double T;
    typedef exprtk::symbol_table<T> symbol_table_t;
    typedef exprtk::expression<T> expression_t;
    typedef exprtk::parser<T> parser_t;

    T u1 = 0.0, u2 = 0.0, u3 = 0.0;

    symbol_table_t symbol_table;
    symbol_table.add_variable("u1", u1);
    symbol_table.add_variable("u2", u2);
    symbol_table.add_variable("u3", u3);

    expression_t expression;
    expression.register_symbol_table(symbol_table);

    parser_t parser;

    // Test 1: integrate 1 over triangle (should be 1)
    std::string expr1 = "1";
    if (parser.compile(expr1, expression)) {
        std::vector<std::string> vars = { "u1", "u2", "u3" };
        T result1 = integrate_simplex(expression, vars, 100);
        std::cout << "Triangle integral of 1: " << result1 << " (expected: 1)"
                  << std::endl;
    }

    // Test 2: integrate u1 over triangle (should be 1/3)
    std::string expr2 = "u1";
    if (parser.compile(expr2, expression)) {
        std::vector<std::string> vars = { "u1", "u2", "u3" };
        T result1 = integrate_simplex(expression, vars, 100);
        std::cout << "Triangle integral of u1: " << result1
                  << " (expected: 0.3333)" << std::endl;
    }

    // Test 3: integrate u1*u2 over triangle (should be 1/12)
    std::string expr3 = "u1*u2";
    if (parser.compile(expr3, expression)) {
        std::vector<std::string> vars = { "u1", "u2", "u3" };
        T result1 = integrate_simplex(expression, vars, 100);
        std::cout << "Triangle integral of u1*u2: " << result1
                  << " (expected: 0.0833)" << std::endl;
    }
}

void test_tetrahedron_integral()
{
    typedef double T;
    typedef exprtk::symbol_table<T> symbol_table_t;
    typedef exprtk::expression<T> expression_t;
    typedef exprtk::parser<T> parser_t;

    T u1 = 0.0, u2 = 0.0, u3 = 0.0, u4 = 0.0;

    symbol_table_t symbol_table;
    symbol_table.add_variable("u1", u1);
    symbol_table.add_variable("u2", u2);
    symbol_table.add_variable("u3", u3);
    symbol_table.add_variable("u4", u4);

    expression_t expression;
    expression.register_symbol_table(symbol_table);

    parser_t parser;

    // Test 1: integrate 1 over tetrahedron (should be 1)
    std::string expr1 = "1";
    if (parser.compile(expr1, expression)) {
        std::vector<std::string> vars = { "u1", "u2", "u3", "u4" };
        T result = integrate_simplex(expression, vars, 100);
        std::cout << "Tetrahedron integral of 1: " << result
                  << " (expected: 1.0)" << std::endl;
    }

    // Test 2: integrate u1 over tetrahedron (should be 1/4)
    std::string expr2 = "u1";
    if (parser.compile(expr2, expression)) {
        std::vector<std::string> vars = { "u1", "u2", "u3", "u4" };
        T result = integrate_simplex(expression, vars, 100);
        std::cout << "Tetrahedron integral of u1: " << result
                  << " (expected: 0.25)" << std::endl;
    }
}

int main()
{
    std::cout << "Testing multi-variable Simpson integration on simplexes:"
              << std::endl;
    std::cout << "======================================================="
              << std::endl;

    test_line_integral();
    std::cout << std::endl;

    test_triangle_integral();
    std::cout << std::endl;

    test_tetrahedron_integral();

    return 0;
}