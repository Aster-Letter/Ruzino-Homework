#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <RZSolver/Solver.hpp>
#include <iostream>

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace Solver {

template<typename EigenSolver>
class EigenDirectSolver : public LinearSolver {
   private:
    std::string solver_name;

   public:
    EigenDirectSolver(const std::string& name) : solver_name(name)
    {
    }

    std::string getName() const override
    {
        return solver_name;
    }
    bool isIterative() const override
    {
        return false;
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

            auto decompose_start = std::chrono::high_resolution_clock::now();
            solver.compute(A);
            auto decompose_end = std::chrono::high_resolution_clock::now();

            result.setup_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    decompose_end - decompose_start);

            if (solver.info() != Eigen::Success) {
                result.error_message = "Matrix decomposition failed";
                return result;
            }

            auto solve_start = std::chrono::high_resolution_clock::now();
            x = solver.solve(b);
            auto solve_end = std::chrono::high_resolution_clock::now();

            result.solve_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    solve_end - solve_start);

            result.converged = (solver.info() == Eigen::Success);
            result.iterations = 1;  // Direct methods solve in one step

            // Check for NaN/infinite results
            if (!x.allFinite()) {
                result.converged = false;
                result.error_message = "Solver produced NaN/infinite values";
                result.final_residual = std::numeric_limits<float>::quiet_NaN();
                return result;
            }

            // Compute relative residual for consistency with iterative solvers
            Eigen::VectorXf residual = A * x - b;
            float b_norm = b.norm();
            result.final_residual =
                (b_norm > 0) ? residual.norm() / b_norm : residual.norm();

            if (config.verbose) {
                std::cout << solver_name << ": decomposition "
                          << result.setup_time.count() << "μs, solve "
                          << result.solve_time.count()
                          << "μs, residual: " << result.final_residual
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
class EigenLUSolver
    : public EigenDirectSolver<Eigen::SparseLU<Eigen::SparseMatrix<float>>> {
   public:
    EigenLUSolver() : EigenDirectSolver("Eigen Sparse LU")
    {
    }
};

class EigenCholeskySolver
    : public EigenDirectSolver<
          Eigen::SimplicialLLT<Eigen::SparseMatrix<float>>> {
   public:
    EigenCholeskySolver() : EigenDirectSolver("Eigen Sparse Cholesky")
    {
    }
};

class EigenQRSolver : public EigenDirectSolver<Eigen::SparseQR<
                          Eigen::SparseMatrix<float>,
                          Eigen::COLAMDOrdering<int>>> {
   public:
    EigenQRSolver() : EigenDirectSolver("Eigen Sparse QR")
    {
    }
};

// Factory functions
std::unique_ptr<LinearSolver> createEigenLUSolver()
{
    return std::make_unique<EigenLUSolver>();
}

std::unique_ptr<LinearSolver> createEigenCholeskySolver()
{
    return std::make_unique<EigenCholeskySolver>();
}

std::unique_ptr<LinearSolver> createEigenQRSolver()
{
    return std::make_unique<EigenQRSolver>();
}

}  // namespace Solver

USTC_CG_NAMESPACE_CLOSE_SCOPE
