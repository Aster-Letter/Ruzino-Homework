#include <RZSolver/Solver.hpp>
#include <unordered_map>
#include <functional>

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace Solver {

// Forward declarations of factory functions
std::unique_ptr<LinearSolver> createCudaCGSolver();
std::unique_ptr<LinearSolver> createEigenCGSolver();
std::unique_ptr<LinearSolver> createEigenBiCGStabSolver();
std::unique_ptr<LinearSolver> createEigenLUSolver();
std::unique_ptr<LinearSolver> createEigenCholeskySolver();
std::unique_ptr<LinearSolver> createEigenQRSolver();

using SolverCreator = std::function<std::unique_ptr<LinearSolver>()>;

static std::unordered_map<SolverType, SolverCreator> solverRegistry = {
    {SolverType::CUDA_CG, createCudaCGSolver},
    {SolverType::EIGEN_ITERATIVE_CG, createEigenCGSolver},
    {SolverType::EIGEN_ITERATIVE_BICGSTAB, createEigenBiCGStabSolver},
    {SolverType::EIGEN_DIRECT_LU, createEigenLUSolver},
    {SolverType::EIGEN_DIRECT_CHOLESKY, createEigenCholeskySolver},
    {SolverType::EIGEN_DIRECT_QR, createEigenQRSolver}
};

static std::unordered_map<SolverType, std::string> solverNames = {
    {SolverType::CUDA_CG, "CUDA Conjugate Gradient"},
    {SolverType::EIGEN_ITERATIVE_CG, "Eigen Conjugate Gradient"},
    {SolverType::EIGEN_ITERATIVE_BICGSTAB, "Eigen BiCGSTAB"},
    {SolverType::EIGEN_DIRECT_LU, "Eigen Sparse LU"},
    {SolverType::EIGEN_DIRECT_CHOLESKY, "Eigen Sparse Cholesky"},
    {SolverType::EIGEN_DIRECT_QR, "Eigen Sparse QR"}
};

std::unique_ptr<LinearSolver> SolverFactory::create(SolverType type) {
    auto it = solverRegistry.find(type);
    if (it != solverRegistry.end()) {
        try {
            return it->second();
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create solver: " + std::string(e.what()));
        }
    }
    throw std::invalid_argument("Unknown solver type");
}

std::vector<SolverType> SolverFactory::getAvailableTypes() {
    std::vector<SolverType> types;
    for (const auto& pair : solverRegistry) {
        types.push_back(pair.first);
    }
    return types;
}

std::string SolverFactory::getTypeName(SolverType type) {
    auto it = solverNames.find(type);
    return (it != solverNames.end()) ? it->second : "Unknown";
}

} // namespace Solver

USTC_CG_NAMESPACE_CLOSE_SCOPE
