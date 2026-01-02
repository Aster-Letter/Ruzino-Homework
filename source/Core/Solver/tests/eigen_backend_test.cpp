#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <RZSolver/Solver.hpp>

using namespace Ruzino::Solver;

class EigenBackendTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        // Create test matrices
        createTridiagonalMatrix(100);
        createPositiveDefiniteMatrix(50);
    }

    void createTridiagonalMatrix(int n)
    {
        tridiag_A.resize(n, n);
        std::vector<Eigen::Triplet<float>> triplets;

        for (int i = 0; i < n; ++i) {
            triplets.push_back(Eigen::Triplet<float>(i, i, 2.0f));
            if (i > 0)
                triplets.push_back(Eigen::Triplet<float>(i, i - 1, -1.0f));
            if (i < n - 1)
                triplets.push_back(Eigen::Triplet<float>(i, i + 1, -1.0f));
        }
        tridiag_A.setFromTriplets(triplets.begin(), triplets.end());
        tridiag_b = Eigen::VectorXf::Ones(n);
    }

    void createPositiveDefiniteMatrix(int n)
    {
        spd_A.resize(n, n);
        std::vector<Eigen::Triplet<float>> triplets;

        // Create a symmetric positive definite matrix
        for (int i = 0; i < n; ++i) {
            triplets.push_back(Eigen::Triplet<float>(i, i, 3.0f));
            if (i > 0) {
                triplets.push_back(Eigen::Triplet<float>(i, i - 1, -1.0f));
                triplets.push_back(Eigen::Triplet<float>(i - 1, i, -1.0f));
            }
        }
        spd_A.setFromTriplets(triplets.begin(), triplets.end());
        spd_b = Eigen::VectorXf::Ones(n);
    }

    Eigen::SparseMatrix<float> tridiag_A, spd_A;
    Eigen::VectorXf tridiag_b, spd_b;
};

TEST_F(EigenBackendTest, IterativeCGSolver)
{
    auto solver = SolverFactory::create(SolverType::EIGEN_ITERATIVE_CG);
    ASSERT_NE(solver, nullptr);

    Eigen::VectorXf x = Eigen::VectorXf::Zero(tridiag_A.rows());
    SolverConfig config;
    config.tolerance = 1e-6f;
    config.verbose = true;

    auto result = solver->solve(tridiag_A, tridiag_b, x, config);

    EXPECT_TRUE(result.converged) << "CG failed: " << result.error_message;
    EXPECT_GT(result.iterations, 0);

    // Verify solution
    Eigen::VectorXf residual = tridiag_A * x - tridiag_b;
    EXPECT_LT(residual.norm(), 1e-4f);
}

TEST_F(EigenBackendTest, IterativeBiCGStabSolver)
{
    auto solver = SolverFactory::create(SolverType::EIGEN_ITERATIVE_BICGSTAB);
    ASSERT_NE(solver, nullptr);

    Eigen::VectorXf x = Eigen::VectorXf::Zero(tridiag_A.rows());
    SolverConfig config;
    config.tolerance = 1e-6f;
    config.verbose = true;

    auto result = solver->solve(tridiag_A, tridiag_b, x, config);

    EXPECT_TRUE(result.converged)
        << "BiCGSTAB failed: " << result.error_message;
    EXPECT_GT(result.iterations, 0);

    // Verify solution - use relative residual for BiCGSTAB
    Eigen::VectorXf residual = tridiag_A * x - tridiag_b;
    float relative_residual = residual.norm() / tridiag_b.norm();
    EXPECT_LT(relative_residual, 1e-3f);  // More relaxed for BiCGSTAB
}

TEST_F(EigenBackendTest, DirectLUSolver)
{
    auto solver = SolverFactory::create(SolverType::EIGEN_DIRECT_LU);
    ASSERT_NE(solver, nullptr);

    Eigen::VectorXf x = Eigen::VectorXf::Zero(tridiag_A.rows());
    SolverConfig config;
    config.verbose = true;

    auto result = solver->solve(tridiag_A, tridiag_b, x, config);

    EXPECT_TRUE(result.converged) << "LU failed: " << result.error_message;
    EXPECT_EQ(result.iterations, 1);  // Direct solver

    // Verify solution - use relative residual
    Eigen::VectorXf residual = tridiag_A * x - tridiag_b;
    float relative_residual = residual.norm() / tridiag_b.norm();
    EXPECT_LT(relative_residual, 1e-3f);  // Relaxed for direct solvers
}

TEST_F(EigenBackendTest, DirectCholeskySolver)
{
    auto solver = SolverFactory::create(SolverType::EIGEN_DIRECT_CHOLESKY);
    ASSERT_NE(solver, nullptr);

    Eigen::VectorXf x = Eigen::VectorXf::Zero(spd_A.rows());
    SolverConfig config;
    config.verbose = true;

    auto result = solver->solve(spd_A, spd_b, x, config);

    EXPECT_TRUE(result.converged)
        << "Cholesky failed: " << result.error_message;
    EXPECT_EQ(result.iterations, 1);

    // Verify solution
    Eigen::VectorXf residual = spd_A * x - spd_b;
    EXPECT_LT(residual.norm(), 1e-4f);
}

TEST_F(EigenBackendTest, DirectQRSolver)
{
    auto solver = SolverFactory::create(SolverType::EIGEN_DIRECT_QR);
    ASSERT_NE(solver, nullptr);

    Eigen::VectorXf x = Eigen::VectorXf::Zero(tridiag_A.rows());
    SolverConfig config;
    config.verbose = true;

    auto result = solver->solve(tridiag_A, tridiag_b, x, config);

    EXPECT_TRUE(result.converged) << "QR failed: " << result.error_message;
    EXPECT_EQ(result.iterations, 1);

    // Verify solution - QR can be less accurate than LU for some matrices
    Eigen::VectorXf residual = tridiag_A * x - tridiag_b;
    float relative_residual = residual.norm() / tridiag_b.norm();
    EXPECT_LT(relative_residual, 5e-3f);  // More relaxed for QR
}

TEST_F(EigenBackendTest, SolverProperties)
{
    // Test iterative solver properties
    auto cg_solver = SolverFactory::create(SolverType::EIGEN_ITERATIVE_CG);
    EXPECT_TRUE(cg_solver->isIterative());
    EXPECT_FALSE(cg_solver->requiresGPU());
    EXPECT_EQ(cg_solver->getName(), "Eigen Conjugate Gradient");

    // Test direct solver properties
    auto lu_solver = SolverFactory::create(SolverType::EIGEN_DIRECT_LU);
    EXPECT_FALSE(lu_solver->isIterative());
    EXPECT_FALSE(lu_solver->requiresGPU());
    EXPECT_EQ(lu_solver->getName(), "Eigen Sparse LU");
}
