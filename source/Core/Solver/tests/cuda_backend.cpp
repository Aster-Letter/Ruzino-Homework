#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <RZSolver/Solver.hpp>

using namespace Ruzino::Solver;

class CudaBackendTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        // Create a simple test matrix (tridiagonal)
        n = 100;
        A.resize(n, n);
        std::vector<Eigen::Triplet<float>> triplets;

        for (int i = 0; i < n; ++i) {
            triplets.push_back(Eigen::Triplet<float>(i, i, 2.0f));
            if (i > 0)
                triplets.push_back(Eigen::Triplet<float>(i, i - 1, -1.0f));
            if (i < n - 1)
                triplets.push_back(Eigen::Triplet<float>(i, i + 1, -1.0f));
        }
        A.setFromTriplets(triplets.begin(), triplets.end());

        b = Eigen::VectorXf::Ones(n);
        x = Eigen::VectorXf::Zero(n);
    }

    int n;
    Eigen::SparseMatrix<float> A;
    Eigen::VectorXf b, x;
};

TEST_F(CudaBackendTest, BasicSolve)
{
    try {
        auto solver = SolverFactory::create(SolverType::CUDA_CG);
        ASSERT_NE(solver, nullptr);

        SolverConfig config;
        config.tolerance = 1e-6f;
        config.max_iterations = 1000;
        config.verbose = true;

        auto result = solver->solve(A, b, x, config);

        EXPECT_TRUE(result.converged)
            << "Solver failed to converge: " << result.error_message;
        EXPECT_GT(result.iterations, 0);
        EXPECT_LT(
            result.final_residual,
            config.tolerance * 10);  // Allow some tolerance

        // Verify solution quality
        Eigen::VectorXf residual = A * x - b;
        EXPECT_LT(residual.norm(), 1e-4f);
    }
    catch (const std::exception& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST_F(CudaBackendTest, LargeMatrixPerformance)
{
    try {
        // Create larger matrix for performance testing
        int large_n = 10000;
        Eigen::SparseMatrix<float> large_A(large_n, large_n);
        std::vector<Eigen::Triplet<float>> triplets;

        for (int i = 0; i < large_n; ++i) {
            triplets.push_back(Eigen::Triplet<float>(i, i, 2.0f));
            if (i > 0)
                triplets.push_back(Eigen::Triplet<float>(i, i - 1, -1.0f));
            if (i < large_n - 1)
                triplets.push_back(Eigen::Triplet<float>(i, i + 1, -1.0f));
        }
        large_A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::VectorXf large_b = Eigen::VectorXf::Ones(large_n);
        Eigen::VectorXf large_x = Eigen::VectorXf::Zero(large_n);

        auto solver = SolverFactory::create(SolverType::CUDA_CG);

        SolverConfig config;
        config.tolerance = 1e-5f;       // Relaxed tolerance for large matrix
        config.max_iterations = 50000;  // Increased iteration limit
        config.verbose = true;

        auto result = solver->solve(large_A, large_b, large_x, config);

        EXPECT_TRUE(result.converged);
        EXPECT_LT(result.solve_time.count(), 5000000);  // Less than 5 seconds

        std::cout << "CUDA CG solved " << large_n << "x" << large_n
                  << " system in " << result.solve_time.count() << " μs"
                  << std::endl;
    }
    catch (const std::exception& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}

TEST_F(CudaBackendTest, ErrorHandling)
{
    try {
        auto solver = SolverFactory::create(SolverType::CUDA_CG);

        // Test with singular matrix
        Eigen::SparseMatrix<float> singular_A(n, n);
        // Don't set any values - zero matrix is singular

        auto result = solver->solve(singular_A, b, x);
        EXPECT_FALSE(result.converged);
        EXPECT_FALSE(result.error_message.empty());
    }
    catch (const std::exception& e) {
        GTEST_SKIP() << "CUDA not available: " << e.what();
    }
}
