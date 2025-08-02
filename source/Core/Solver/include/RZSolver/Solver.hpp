#pragma once

#include "api.h"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <memory>
#include <string>
#include <chrono>

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace Solver {

enum class SolverType {
    CUDA_CG,
    EIGEN_ITERATIVE_CG,
    EIGEN_ITERATIVE_BICGSTAB,
    EIGEN_DIRECT_LU,
    EIGEN_DIRECT_CHOLESKY,
    EIGEN_DIRECT_QR
};

struct SolverConfig {
    float tolerance = 1e-6f;
    int max_iterations = 1000;
    bool use_preconditioner = true;
    bool verbose = false;
};

struct SolverResult {
    bool converged = false;
    int iterations = 0;
    float final_residual = 0.0f;
    std::chrono::microseconds solve_time{0};
    std::chrono::microseconds setup_time{0};
    std::string error_message;
};

class RZSOLVER_API LinearSolver {
public:
    virtual ~LinearSolver() = default;
    
    virtual SolverResult solve(
        const Eigen::SparseMatrix<float>& A,
        const Eigen::VectorXf& b,
        Eigen::VectorXf& x,
        const SolverConfig& config = SolverConfig{}
    ) = 0;
    
    virtual std::string getName() const = 0;
    virtual bool isIterative() const = 0;
    virtual bool requiresGPU() const = 0;
};

class RZSOLVER_API SolverFactory {
public:
    static std::unique_ptr<LinearSolver> create(SolverType type);
    static std::vector<SolverType> getAvailableTypes();
    static std::string getTypeName(SolverType type);
};

} // namespace Solver

USTC_CG_NAMESPACE_CLOSE_SCOPE
