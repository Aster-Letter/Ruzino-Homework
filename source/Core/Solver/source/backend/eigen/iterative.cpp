#include <Eigen/IterativeLinearSolvers>
#include <RZSolver/Solver.hpp>
#include <iostream>

USTC_CG_NAMESPACE_OPEN_SCOPE

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
                result.error_message = "Solver produced NaN/infinite values - numerical breakdown";
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
                result.error_message = "Solver produced poor quality solution (residual > 10%)";
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
};

class EigenBiCGStabSolver
    : public EigenIterativeSolver<Eigen::BiCGSTAB<Eigen::SparseMatrix<float>>> {
   public:
    EigenBiCGStabSolver() : EigenIterativeSolver("Eigen BiCGSTAB")
    {
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

USTC_CG_NAMESPACE_CLOSE_SCOPE

