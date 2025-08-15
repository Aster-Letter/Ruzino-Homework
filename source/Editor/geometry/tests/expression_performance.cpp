#include <gtest/gtest.h>

#include <chrono>
#include <string>

#include "../../geometry_nodes/fem_bem/Expression.hpp"

using namespace USTC_CG::fem_bem;

// Test Expression class specific functionality - Derivative interface
int main()
{
    ExpressionD expr2("x + y");
    ExpressionD element1("u+v");
    ExpressionD element2("(u-v)^2");
    ExpressionD compound(expr2, { { "x", element1 }, { "y", element2 } });

    auto derivative = compound.derivative("u");  // 1 + 2 * (u - v)

    // Test that derivative can be used in compound expressions
    auto compound2 = ExpressionD(
        expr2,
        { { "x", element1 }, { "y", derivative } });  // u + v + 1 + 2 * (u - v)

    auto start = std::chrono::high_resolution_clock::now();
    // baseline: 0.002f, 0.002f, 3917 ms.

    long long operations = 0;

#ifdef DEBUG
    auto step = 0.001f;
#else
    auto step = 0.0001f;  // Use smaller step for performance testing
#endif


    for (float u = 0.0f; u <= 1.0f; u += step) {
        for (float v = 0.0f; v <= 1.0f; v += 0.002f) {
            auto eval_at_uv = compound2.evaluate_at({ { "u", u }, { "v", v } });
            operations += 1;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "Evaluation took " << duration << " ms." << std::endl;
    std::cout << "Total operations: " << operations << std::endl;

    std::cout << "flops: " << (operations / (duration / 1000.0))/ 1000.0 << " kflops/s."
              << std::endl;

    // Baseline: 46.32 kflops/s.
    // Unchecked get_variable(): 71.36 kflops/s
    // Use local buffer: 200 kflops/s
    // Release version: 3174.06 kflops/s
 }
