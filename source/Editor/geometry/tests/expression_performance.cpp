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

    for (float u = 0.0f; u <= 1.0f; u += 0.002f) {
        for (float v = 0.0f; v <= 1.0f; v += 0.005f) {
            auto eval_at_uv = compound2.evaluate_at({ { "u", u }, { "v", v } });
            EXPECT_NEAR(
                eval_at_uv,
                3 * u - v + 1,
                1e-6);  // Check against expected linear form
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "Evaluation took " << duration << " ms." << std::endl;
}
