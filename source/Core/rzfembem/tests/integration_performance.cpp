#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "fem_bem/ElementBasis.hpp"
#include "fem_bem/Expression.hpp"

using namespace USTC_CG::fem_bem;

class IntegrationPerformanceTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        // Setup common test data
        setup_2d_triangle();
        setup_3d_tetrahedron();
    }

    void setup_2d_triangle()
    {
        triangle_2d = { pxr::GfVec2d(0.0, 0.0),
                        pxr::GfVec2d(1.0, 0.0),
                        pxr::GfVec2d(0.0, 1.0) };
    }

    void setup_3d_tetrahedron()
    {
        tetrahedron_3d = { pxr::GfVec3d(0.0, 0.0, 0.0),
                           pxr::GfVec3d(1.0, 0.0, 0.0),
                           pxr::GfVec3d(0.0, 1.0, 0.0),
                           pxr::GfVec3d(0.0, 0.0, 1.0) };
    }

    template<typename Func>
    double benchmark_function(const std::string& test_name, Func&& func)
    {
        auto start = std::chrono::high_resolution_clock::now();
        int iterations = 20;
        for (int i = 0; i < iterations; ++i) {
            func();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();

        double avg_time = static_cast<double>(duration) / iterations;
        std::cout << test_name << ": " << avg_time << " μs per operation ("
                  << iterations << " iterations)" << std::endl;

        return avg_time;
    }

    std::vector<pxr::GfVec2d> triangle_2d;
    std::vector<pxr::GfVec3d> tetrahedron_3d;
};

// Test simple expression integration performance
TEST_F(IntegrationPerformanceTest, SimpleExpressionIntegration)
{
    std::cout << "\n=== Simple Expression Integration Performance ==="
              << std::endl;

    Expression simple_expr("1");
    std::vector<const char*> vars = { "u1", "u2", "u3" };

    // Test with different integration intervals
    std::vector<int> intervals = { 10, 50, 100, 200 };

    for (int interval : intervals) {
        auto test_name =
            "Constant_integration_" + std::to_string(interval) + "_intervals";
        benchmark_function(test_name, [&]() {
            integrate_over_simplex(simple_expr, vars, nullptr, interval);
        });
    }
}

// Test polynomial expression performance
TEST_F(IntegrationPerformanceTest, PolynomialExpressionIntegration)
{
    std::cout << "\n=== Polynomial Expression Integration Performance ==="
              << std::endl;

    std::vector<const char*> expressions = { "u1",
                                             "u1 + u2",
                                             "u1*u2",
                                             "u1*u2 + u2*u3",
                                             "u1*u1 + u2*u2 + u3*u3",
                                             "u1*u1*u2 + u2*u2*u3 + u3*u3*u1",
                                             "u1*u1*u1 + u2*u2*u2 + u3*u3*u3" };

    std::vector<const char*> vars = { "u1", "u2", "u3" };

    for (const auto& expr_str : expressions) {
        Expression expr(expr_str);  // Add variable list
        auto test_name = std::string("Polynomial_") + expr_str;
        benchmark_function(test_name, [&]() {
            integrate_over_simplex(expr, vars, nullptr, 50);
        });
    }
}

// Test compound expression performance
TEST_F(IntegrationPerformanceTest, CompoundExpressionIntegration)
{
    std::cout << "\n=== Compound Expression Integration Performance ==="
              << std::endl;

    Expression outer("x + y");
    Expression element1("u1 + u2");
    Expression element2("u1*u2 + u2*u3");
    Expression compound(outer, { { "x", element1 }, { "y", element2 } });

    std::vector<const char*> vars = { "u1", "u2", "u3" };

    benchmark_function("Simple_compound_expression", [&]() {
        integrate_over_simplex(compound, vars, nullptr, 50);
    });

    // More complex compound expression
    Expression complex_outer("a*b + c*d");
    Expression elem_a("u1*u1");
    Expression elem_b("u2*u2");
    Expression elem_c("u3*u3");
    Expression elem_d("u1*u2*u3");
    Expression complex_compound(
        complex_outer,
        { { "a", elem_a }, { "b", elem_b }, { "c", elem_c }, { "d", elem_d } });

    benchmark_function("Complex_compound_expression", [&]() {
        integrate_over_simplex(complex_compound, vars, nullptr, 50);
    });
}

// Test coordinate mapping performance
TEST_F(IntegrationPerformanceTest, CoordinateMappingIntegration)
{
    std::cout << "\n=== Coordinate Mapping Integration Performance ==="
              << std::endl;

    auto fem2d = make_fem_2d();
    auto mapping = fem2d->create_coordinate_mapping(triangle_2d);

    std::vector<const char*> spatial_expressions = {
        "x", "y", "x + y", "x*y", "x*x + y*y", "x*x*y + y*y*x"
    };

    std::vector<const char*> vars = { "u1", "u2", "u3" };

    for (const auto& expr_str : spatial_expressions) {
        Expression expr(expr_str);
        auto test_name = std::string("Mapping_") + expr_str;
        benchmark_function(test_name, [&]() {
            integrate_over_simplex(expr, vars, mapping, 50);
        });
    }
}

// Test derivative integration performance
TEST_F(IntegrationPerformanceTest, DerivativeIntegration)
{
    std::cout << "\n=== Derivative Integration Performance ===" << std::endl;

    Expression base_expr("u1*u1 + u2*u2 + u3*u3");  // Add variable list
    std::vector vars = { "u1", "u2", "u3" };
    // Test derivative computation and integration
    benchmark_function("Base_expression_integration", [&]() {
        integrate_over_simplex(base_expr, vars, nullptr, 10);
    });

    auto du1 = base_expr.derivative("u1");
    benchmark_function("Derivative_u1_integration", [&]() {
        integrate_over_simplex(du1, vars, nullptr, 10);
    });

    auto du2 = base_expr.derivative("u2");
    benchmark_function("Derivative_u2_integration", [&]() {
        integrate_over_simplex(du2, vars, nullptr, 10);
    });

    // Test gradient integration
    auto gradient = base_expr.gradient({ "u1", "u2" });
    benchmark_function("Gradient_component_0_integration", [&]() {
        integrate_over_simplex(gradient[0], vars, nullptr, 10);
    });

    benchmark_function("Gradient_component_1_integration", [&]() {
        integrate_over_simplex(gradient[1], vars, nullptr, 10);
    });
}

// Test element basis integration performance
TEST_F(IntegrationPerformanceTest, ElementBasisIntegration)
{
    std::cout << "\n=== Element Basis Integration Performance ===" << std::endl;

    // Test different element types
    auto fem1d = make_fem_1d();
    fem1d->add_vertex_expression("u1");
    fem1d->add_vertex_expression("1 - u1");

    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("u1");
    fem2d->add_vertex_expression("u2");
    fem2d->add_vertex_expression("1 - u1 - u2");
    fem2d->add_edge_expression("u1*u2");

    auto fem3d = make_fem_3d();
    fem3d->add_vertex_expression("u1");
    fem3d->add_vertex_expression("u2");
    fem3d->add_vertex_expression("u3");
    fem3d->add_vertex_expression("1 - u1 - u2 - u3");
    fem3d->add_volume_expression("u1*u2*u3");

    // Test vertex expression integration
    Expression test_expr("1");

    benchmark_function("FEM1D_vertex_integration", [&]() {
        const auto& vertex_exprs = fem1d->get_vertex_expressions();
        for (const auto& shape_func : vertex_exprs) {
            Expression product = shape_func * test_expr;
            integrate_over_simplex(
                product, fem1d->get_barycentric_names(), nullptr, 50);
        }
    });

    benchmark_function("FEM2D_vertex_integration", [&]() {
        const auto& vertex_exprs = fem2d->get_vertex_expressions();
        for (const auto& shape_func : vertex_exprs) {
            Expression product = shape_func * test_expr;
            integrate_over_simplex(
                product, fem2d->get_barycentric_names(), nullptr, 50);
        }
    });

    benchmark_function("FEM3D_vertex_integration", [&]() {
        const auto& vertex_exprs = fem3d->get_vertex_expressions();
        for (const auto& shape_func : vertex_exprs) {
            Expression product = shape_func * test_expr;
            integrate_over_simplex(
                product, fem3d->get_barycentric_names(), nullptr, 30);
        }
    });
}

// Test arithmetic operations performance
TEST_F(IntegrationPerformanceTest, ArithmeticOperationsIntegration)
{
    std::cout << "\n=== Arithmetic Operations Integration Performance ==="
              << std::endl;

    Expression expr1("u1 + 2");
    Expression expr2("u2 * 3");
    std::vector vars = { "u1", "u2", "u3" };
    benchmark_function("Addition_integration", [&]() {
        auto sum = expr1 + expr2;
        integrate_over_simplex(sum, vars, nullptr, 50);
    });

    benchmark_function("Subtraction_integration", [&]() {
        auto diff = expr1 - expr2;
        integrate_over_simplex(diff, vars, nullptr, 50);
    });

    benchmark_function("Multiplication_integration", [&]() {
        auto prod = expr1 * expr2;
        integrate_over_simplex(prod, vars, nullptr, 50);
    });

    benchmark_function("Division_integration", [&]() {
        auto quot = expr1 / expr2;
        integrate_over_simplex(quot, vars, nullptr, 50);
    });

    // Complex arithmetic chain
    benchmark_function("Complex_arithmetic_integration", [&]() {
        auto complex_expr = (expr1 + expr2) * (expr1 - expr2) /
                            (expr1 + Expression::constant(1.0));
        integrate_over_simplex(complex_expr, vars, nullptr, 50);
    });
}

// Test scaling with integration intervals
TEST_F(IntegrationPerformanceTest, IntegrationIntervalScaling)
{
    std::cout << "\n=== Integration Interval Scaling Performance ==="
              << std::endl;

    Expression test_expr("u1*u2 + u2*u3 + u3*u1");  // Add variable list
    std::vector vars = { "u1", "u2", "u3" };

    std::vector<int> intervals = { 5, 10, 20, 30 };

    for (int interval : intervals) {
        auto test_name = "Intervals_" + std::to_string(interval);
        double avg_time = benchmark_function(test_name, [&]() {
            integrate_over_simplex(test_expr, vars, nullptr, interval);
        });

        // Calculate approximate operations per second
        double ops_per_sec = 1000000.0 / avg_time;  // Convert μs to ops/sec
        std::cout << "  -> " << static_cast<int>(ops_per_sec)
                  << " integrations/second" << std::endl;
    }
}

// Test memory allocation performance
TEST_F(IntegrationPerformanceTest, MemoryAllocationPerformance)
{
    std::cout << "\n=== Memory Allocation Performance ===" << std::endl;

    std::vector vars = { "u1", "u2", "u3" };

    // Tevector<const char*>ion creation and integration
    benchmark_function("Expression_creation_and_integration", [&]() {
        Expression temp_expr("u1*u1 + u2*u2");  // Add variable list
        integrate_over_simplex(temp_expr, vars, nullptr, 20);
    });

    // Test compound expression creation
    benchmark_function("Compound_expression_creation", [&]() {
        Expression outer("x + y");
        Expression inner1("u1");  // Add variable list
        Expression inner2("u2");  // Add variable list
        Expression compound(outer, { { "x", inner1 }, { "y", inner2 } });
        integrate_over_simplex(compound, vars, nullptr, 20);
    });

    // Test arithmetic operations memory usage
    benchmark_function("Arithmetic_operations_memory", [&]() {
        Expression e1("u1");  // Add variable list
        Expression e2("u2");  // Add variable list
        auto result = e1 + e2 * Expression::constant(3.0) - e1 / e2;
        integrate_over_simplex(result, vars, nullptr, 20);
    });
}

// Main function for standalone performance testing
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "Integration Performance Test Suite" << std::endl;
    std::cout << "===================================" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    int result = RUN_ALL_TESTS();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end_time - start_time)
                              .count();

    std::cout << "\nTotal test suite execution time: " << total_duration
              << " ms" << std::endl;
    std::cout << "insert_or_assign() is called " << g_insert_or_assign_calls
              << " times." << std::endl;
    std::cout << "insert_unchecked() is called " << g_insert_unchecked_calls
              << " times." << std::endl;
    std::cout << "evaluate_at() is called " << g_evaluate_calls << " times."
              << std::endl;

    return result;
}
