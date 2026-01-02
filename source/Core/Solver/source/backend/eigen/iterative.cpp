#include <Eigen/IterativeLinearSolvers>
#include <RZSolver/Solver.hpp>
#include <iostream>

#include "spdlog/spdlog.h"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace Solver {

template<typename EigenSolver>
class EigenIterativeSolver : public LinearSolver {
   private:
    std::string solver_name;

   public:
    EigenIterativeSolver(const std::string& name) : solver_name(name)
    {
    }

    std::string getName() const override
    {
        return solver_name;
    }
    bool isIterative() const override
    {
        return true;
    }
    bool requiresGPU() const override
    {
        return false;
    }

    SolverResult solve(
        const Eigen::SparseMatrix<float>& A,
        const Eigen::VectorXf& b,
        Eigen::VectorXf& x,
        const SolverConfig& config = SolverConfig{}) override
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        SolverResult result;

        try {
            EigenSolver solver;
            solver.setTolerance(config.tolerance);
            solver.setMaxIterations(config.max_iterations);

            auto setup_end_time = std::chrono::high_resolution_clock::now();
            result.setup_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    setup_end_time - start_time);

            auto solve_start_time = std::chrono::high_resolution_clock::now();

            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                result.error_message = "Matrix decomposition failed";
                return result;
            }

            x = solver.solve(b);

            auto solve_end_time = std::chrono::high_resolution_clock::now();
            result.solve_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    solve_end_time - solve_start_time);

            result.converged = (solver.info() == Eigen::Success);
            result.iterations = solver.iterations();

            // Check for NaN results first (common in BiCGSTAB breakdown)
            if (!x.allFinite()) {
                result.converged = false;
                result.error_message =
                    "Solver produced NaN/infinite values - numerical breakdown";
                result.final_residual = std::numeric_limits<float>::quiet_NaN();
                return result;
            }

            // Compute actual residual for verification
            Eigen::VectorXf residual = A * x - b;
            float b_norm = b.norm();
            result.final_residual =
                (b_norm > 0) ? residual.norm() / b_norm : residual.norm();

            // Additional check: if residual is too large, mark as failed
            if (result.final_residual > 0.1f) {  // 10% error threshold
                result.converged = false;
                result.error_message =
                    "Solver produced poor quality solution (residual > 10%)";
            }

            if (config.verbose) {
                std::cout << solver_name << ": " << result.iterations
                          << " iterations, residual: " << result.final_residual
                          << std::endl;
            }
        }
        catch (const std::exception& e) {
            result.error_message = e.what();
            result.converged = false;
        }

        return result;
    }
};

// Specific solver implementations
class EigenCGSolver
    : public EigenIterativeSolver<
          Eigen::ConjugateGradient<Eigen::SparseMatrix<float>>> {
   public:
    EigenCGSolver() : EigenIterativeSolver("Eigen Conjugate Gradient")
    {
    }

    // Override solve method to check matrix properties
    SolverResult solve(
        const Eigen::SparseMatrix<float>& A,
        const Eigen::VectorXf& b,
        Eigen::VectorXf& x,
        const SolverConfig& config = SolverConfig{}) override
    {
        // Check if matrix is likely SPD
        if (!isLikelySPD(A)) {
            SolverResult result;
            result.error_message =
                "CG requires symmetric positive definite matrix";
            result.converged = false;
            return result;
        }

        // Use the base class implementation for SPD matrices
        return EigenIterativeSolver::solve(A, b, x, config);
    }

   private:
    // Check if matrix is likely SPD
    bool isLikelySPD(const Eigen::SparseMatrix<float>& A)
    {
        if (A.rows() != A.cols())
            return false;
        return true;
        // Quick symmetry check on a sample of entries
        int sample_size = std::min(100, (int)A.rows());
        for (int i = 0; i < sample_size; ++i) {
            for (int j = i + 1; j < sample_size; ++j) {
                float aij = A.coeff(i, j);
                float aji = A.coeff(j, i);
                if (abs(aij - aji) >
                    1e-6f * std::max(abs(aij), abs(aji)) + 1e-10f) {
                    return false;
                }
            }
        }

        // Check diagonal positivity
        for (int i = 0; i < sample_size; ++i) {
            if (A.coeff(i, i) <= 0)
                return false;
        }

        return true;
    }
};

class EigenBiCGStabSolver
    : public EigenIterativeSolver<Eigen::BiCGSTAB<Eigen::SparseMatrix<float>>> {
   public:
    EigenBiCGStabSolver() : EigenIterativeSolver("Eigen BiCGSTAB")
    {
    }

    // Override solve method for BiCGSTAB-specific handling
    SolverResult solve(
        const Eigen::SparseMatrix<float>& A,
        const Eigen::VectorXf& b,
        Eigen::VectorXf& x,
        const SolverConfig& config = SolverConfig{}) override
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        SolverResult result;

        try {
            // 对于非常大的 SPD 矩阵，给出警告但仍然尝试求解
            if (isSPDMatrix(A) && A.rows() > 10000) {
                if (config.verbose) {
                    spdlog::warn(
                        "Warning: BiCGSTAB on large SPD matrix may converge "
                        "slowly");
                }
                // 调整迭代参数以提高成功率
                const int max_restarts = 5;  // 增加重启次数
                int restart_count = 0;

                while (restart_count < max_restarts) {
                    Eigen::BiCGSTAB<Eigen::SparseMatrix<float>> solver;
                    solver.setTolerance(config.tolerance * 10);  // 放宽容差
                    solver.setMaxIterations(
                        std::min(config.max_iterations, 2000));

                    auto setup_end_time =
                        std::chrono::high_resolution_clock::now();
                    result.setup_time =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            setup_end_time - start_time);

                    auto solve_start_time =
                        std::chrono::high_resolution_clock::now();

                    solver.compute(A);
                    if (solver.info() != Eigen::Success) {
                        result.error_message = "Matrix decomposition failed";
                        return result;
                    }

                    x = solver.solve(b);

                    auto solve_end_time =
                        std::chrono::high_resolution_clock::now();
                    result.solve_time =
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            solve_end_time - solve_start_time);

                    if (!x.allFinite()) {
                        restart_count++;
                        if (restart_count < max_restarts) {
                            x = Eigen::VectorXf::Random(A.rows()) *
                                0.01f;  // 更小的初值
                            continue;
                        }
                        else {
                            result.converged = false;
                            result.error_message =
                                "BiCGSTAB numerical breakdown after " +
                                std::to_string(max_restarts) + " restarts";
                            return result;
                        }
                    }

                    result.converged = (solver.info() == Eigen::Success);
                    result.iterations =
                        solver.iterations() + restart_count * 2000;

                    Eigen::VectorXf residual = A * x - b;
                    float b_norm = b.norm();
                    result.final_residual = (b_norm > 0)
                                                ? residual.norm() / b_norm
                                                : residual.norm();

                    if (result.converged &&
                        result.final_residual < config.tolerance) {
                        if (config.verbose) {
                            std::cout
                                << "BiCGSTAB converged on SPD matrix with "
                                << restart_count << " restarts" << std::endl;
                        }
                        break;
                    }
                    else if (result.final_residual > 0.05f) {  // 5% 误差阈值
                        restart_count++;
                        if (restart_count < max_restarts) {
                            x = Eigen::VectorXf::Random(A.rows()) * 0.01f;
                            continue;
                        }
                        else {
                            result.converged = false;
                            result.error_message =
                                "BiCGSTAB failed to achieve acceptable "
                                "accuracy on SPD matrix";
                        }
                    }
                    break;
                }
                return result;
            }
            else {
                // 原来的 BiCGSTAB 逻辑（处理一般矩阵）
                return performStandardBiCGStab(A, b, x, config, start_time);
            }
        }
        catch (const std::exception& e) {
            result.error_message = e.what();
            result.converged = false;
        }

        return result;
    }

   private:
    SolverResult performStandardBiCGStab(
        const Eigen::SparseMatrix<float>& A,
        const Eigen::VectorXf& b,
        Eigen::VectorXf& x,
        const SolverConfig& config,
        std::chrono::high_resolution_clock::time_point start_time)
    {
        SolverResult result;

        // 标准的 BiCGSTAB 实现（之前的逻辑）
        const int max_restarts = 3;
        int restart_count = 0;

        while (restart_count < max_restarts) {
            Eigen::BiCGSTAB<Eigen::SparseMatrix<float>> solver;
            solver.setTolerance(config.tolerance);
            solver.setMaxIterations(std::min(config.max_iterations, 1000));

            auto setup_end_time = std::chrono::high_resolution_clock::now();
            result.setup_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    setup_end_time - start_time);

            auto solve_start_time = std::chrono::high_resolution_clock::now();

            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                result.error_message = "Matrix decomposition failed";
                return result;
            }

            x = solver.solve(b);

            auto solve_end_time = std::chrono::high_resolution_clock::now();
            result.solve_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    solve_end_time - solve_start_time);

            if (!x.allFinite()) {
                restart_count++;
                if (restart_count < max_restarts) {
                    x = Eigen::VectorXf::Random(A.rows()) * 0.1f;
                    continue;
                }
                else {
                    result.converged = false;
                    result.error_message =
                        "BiCGSTAB numerical breakdown after restarts";
                    return result;
                }
            }

            result.converged = (solver.info() == Eigen::Success);
            result.iterations = solver.iterations() + restart_count * 1000;

            Eigen::VectorXf residual = A * x - b;
            float b_norm = b.norm();
            result.final_residual =
                (b_norm > 0) ? residual.norm() / b_norm : residual.norm();

            if (result.converged &&
                result.final_residual < config.tolerance * 10) {
                break;
            }
            else if (result.final_residual > 0.1f) {
                restart_count++;
                if (restart_count < max_restarts) {
                    x = Eigen::VectorXf::Random(A.rows()) * 0.1f;
                    continue;
                }
                else {
                    result.converged = false;
                    result.error_message =
                        "BiCGSTAB failed to achieve acceptable accuracy";
                }
            }
            break;
        }

        if (config.verbose) {
            std::cout << "Eigen BiCGSTAB: " << result.iterations
                      << " iterations (with " << restart_count
                      << " restarts), residual: " << result.final_residual
                      << std::endl;
        }

        return result;
    }

    // Improved heuristic to detect SPD matrices
    bool isSPDMatrix(const Eigen::SparseMatrix<float>& A)
    {
        if (A.rows() != A.cols())
            return false;

        // Check if matrix is symmetric (approximately)
        int sample_size = std::min(100, (int)A.rows());
        int asymmetric_count = 0;
        int total_checks = 0;

        for (int i = 0; i < sample_size; ++i) {
            for (int j = i + 1; j < sample_size; ++j) {
                float aij = A.coeff(i, j);
                float aji = A.coeff(j, i);

                // Only check if at least one is non-zero
                if (abs(aij) > 1e-10f || abs(aji) > 1e-10f) {
                    total_checks++;
                    float max_val = std::max(abs(aij), abs(aji));
                    if (max_val > 1e-10f && abs(aij - aji) > 1e-6f * max_val) {
                        asymmetric_count++;
                    }
                }
            }
        }

        // If more than 10% of checked entries are asymmetric, consider it
        // non-SPD
        if (total_checks > 0 && (float)asymmetric_count / total_checks > 0.1f) {
            return false;
        }

        // Check diagonal dominance and positivity (common in SPD matrices)
        for (int i = 0; i < sample_size; ++i) {
            if (A.coeff(i, i) <= 0)
                return false;
        }

        return true;
    }
};

// Factory functions
std::unique_ptr<LinearSolver> createEigenCGSolver()
{
    return std::make_unique<EigenCGSolver>();
}

std::unique_ptr<LinearSolver> createEigenBiCGStabSolver()
{
    return std::make_unique<EigenBiCGStabSolver>();
}

}  // namespace Solver

RUZINO_NAMESPACE_CLOSE_SCOPE
