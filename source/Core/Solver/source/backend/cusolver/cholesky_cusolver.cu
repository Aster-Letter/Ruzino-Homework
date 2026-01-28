#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <RHI/cuda.hpp>
#include <RHI/internal/cuda_extension.hpp>
#include <RZSolver/Solver.hpp>
#include <iostream>

RUZINO_NAMESPACE_OPEN_SCOPE

namespace Solver {

class CuSolverCholeskySolver : public LinearSolver {
   private:
    cusolverSpHandle_t cusolverSpHandle;
    cusolverDnHandle_t cusolverDnHandle;
    cusparseHandle_t cusparseHandle;
    bool initialized = false;

    // Cached buffers for solve() method (Eigen input)
    // We cache these to avoid reallocation when solve() is called repeatedly
    // with same size matrix
    int cached_nnz = 0;
    int cached_n = 0;
    Ruzino::cuda::CUDALinearBufferHandle d_csrVal_cached;
    Ruzino::cuda::CUDALinearBufferHandle d_csrRowPtr_cached;
    Ruzino::cuda::CUDALinearBufferHandle d_csrColInd_cached;
    Ruzino::cuda::CUDALinearBufferHandle d_b_cached;
    Ruzino::cuda::CUDALinearBufferHandle d_x_cached;

    // Cached buffers for solveDenseGPU() method
    int cached_dense_n = 0;
    int cached_lwork = 0;
    float* d_work_cached = nullptr;
    int* d_info_cached = nullptr;
    float* d_A_copy_cached = nullptr;

   public:
    CuSolverCholeskySolver()
    {
        if (cusolverSpCreate(&cusolverSpHandle) != CUSOLVER_STATUS_SUCCESS ||
            cusolverDnCreate(&cusolverDnHandle) != CUSOLVER_STATUS_SUCCESS ||
            cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error(
                "Failed to create cuSOLVER/cuSPARSE handles");
        }
        initialized = true;
    }

    ~CuSolverCholeskySolver()
    {
        if (initialized) {
            cusolverSpDestroy(cusolverSpHandle);
            cusolverDnDestroy(cusolverDnHandle);
            cusparseDestroy(cusparseHandle);

            // Free cached dense solver buffers
            if (d_work_cached)
                cudaFree(d_work_cached);
            if (d_info_cached)
                cudaFree(d_info_cached);
            if (d_A_copy_cached)
                cudaFree(d_A_copy_cached);
        }
    }

    std::string getName() const override
    {
        return "cuSOLVER Cholesky (Direct)";
    }

    bool isIterative() const override
    {
        return false;  // Direct solver
    }

    bool requiresGPU() const override
    {
        return true;
    }

    // Direct GPU interface implementation
    SolverResult solveGPU(
        int n,
        int nnz,
        const int* d_row_offsets,
        const int* d_col_indices,
        const float* d_values,
        const float* d_b,
        float* d_x,
        const SolverConfig& config = SolverConfig{}) override
    {
        SolverResult result;
        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            // Create cuSOLVER matrix descriptor
            // For Cholesky, matrix must be symmetric positive definite
            cusparseMatDescr_t descrA;
            cusparseCreateMatDescr(&descrA);
            cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

            int singularity = 0;

            // Call cuSOLVER Cholesky solver
            // Note: cusolverSpScsrlsvchol solves A*x = b where A is SPD
            // The matriSpx should already be symmetric from the Hessian
            // construction
            cusolverStatus_t status = cusolverSpScsrlsvchol(
                cusolverSpHandle,
                n,
                nnz,
                descrA,
                d_values,
                d_row_offsets,
                d_col_indices,
                d_b,
                config.tolerance,
                0,  // no reordering for now
                d_x,
                &singularity);

            cusparseDestroyMatDescr(descrA);

            if (status != CUSOLVER_STATUS_SUCCESS) {
                result.converged = false;
                result.error_message = "cuSOLVER Cholesky failed with status " +
                                       std::to_string(status);
                if (config.verbose) {
                    std::cout << result.error_message << std::endl;
                }
            }
            else if (singularity >= 0) {
                result.converged = false;
                result.error_message =
                    "Matrix is not positive definite (singularity at " +
                    std::to_string(singularity) + ")";
                if (config.verbose) {
                    std::cout << result.error_message << std::endl;
                }
            }
            else {
                result.converged = true;
                result.iterations = 1;  // Direct solver, single "iteration"
                result.final_residual = 0.0f;  // Direct solver

                if (config.verbose) {
                    std::cout << "cuSOLVER Cholesky direct solve completed "
                                 "successfully"
                              << std::endl;
                }
            }
        }
        catch (const std::exception& e) {
            result.converged = false;
            result.error_message =
                std::string("cuSOLVER Cholesky error: ") + e.what();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.solve_time =
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);

        return result;
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
            int n = A.rows();
            int nnz = A.nonZeros();

            // Reallocate cached buffers if size changed
            if (n != cached_n || nnz != cached_nnz || !d_csrVal_cached) {
                cached_n = n;
                cached_nnz = nnz;

                Ruzino::cuda::CUDALinearBufferDesc val_desc, rowptr_desc,
                    colind_desc, vec_desc;
                val_desc.element_count = nnz;
                val_desc.element_size = sizeof(float);
                rowptr_desc.element_count = n + 1;
                rowptr_desc.element_size = sizeof(int);
                colind_desc.element_count = nnz;
                colind_desc.element_size = sizeof(int);
                vec_desc.element_count = n;
                vec_desc.element_size = sizeof(float);

                d_csrVal_cached =
                    Ruzino::cuda::create_cuda_linear_buffer(val_desc);
                d_csrRowPtr_cached =
                    Ruzino::cuda::create_cuda_linear_buffer(rowptr_desc);
                d_csrColInd_cached =
                    Ruzino::cuda::create_cuda_linear_buffer(colind_desc);
                d_b_cached = Ruzino::cuda::create_cuda_linear_buffer(vec_desc);
                d_x_cached = Ruzino::cuda::create_cuda_linear_buffer(vec_desc);
            }

            // Convert to CSR format on host
            Eigen::SparseMatrix<float, Eigen::RowMajor> A_csr = A;

            std::vector<float> csrVal(nnz);
            std::vector<int> csrRowPtr(n + 1);
            std::vector<int> csrColInd(nnz);

            // Copy CSR data
            int idx = 0;
            csrRowPtr[0] = 0;
            for (int i = 0; i < n; ++i) {
                for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator
                         it(A_csr, i);
                     it;
                     ++it) {
                    csrVal[idx] = it.value();
                    csrColInd[idx] = it.col();
                    idx++;
                }
                csrRowPtr[i + 1] = idx;
            }

            // Debug output
            if (config.verbose) {
                std::cout << "Matrix size: " << n << "x" << n << std::endl;
                std::cout << "Total nnz: " << nnz << std::endl;

                // Check symmetry (sampling)
                int sample_size = std::min(100, n);
                bool symmetric = true;
                for (int i = 0; i < sample_size && symmetric; ++i) {
                    for (int j = i + 1; j < sample_size; ++j) {
                        float aij = A.coeff(i, j);
                        float aji = A.coeff(j, i);
                        if (std::abs(aij - aji) >
                            1e-6f * std::max(std::abs(aij), std::abs(aji))) {
                            symmetric = false;
                            std::cout
                                << "Warning: Matrix may not be symmetric at ("
                                << i << "," << j << "): " << aij << " vs "
                                << aji << std::endl;
                            break;
                        }
                    }
                }
                if (symmetric) {
                    std::cout << "Matrix appears symmetric (sample check)"
                              << std::endl;
                }
            }

            // Copy to device (using cached buffers)
            cudaMemcpy(
                (void*)d_csrVal_cached->get_device_ptr(),
                csrVal.data(),
                nnz * sizeof(float),
                cudaMemcpyHostToDevice);
            cudaMemcpy(
                (void*)d_csrRowPtr_cached->get_device_ptr(),
                csrRowPtr.data(),
                (n + 1) * sizeof(int),
                cudaMemcpyHostToDevice);
            cudaMemcpy(
                (void*)d_csrColInd_cached->get_device_ptr(),
                csrColInd.data(),
                nnz * sizeof(int),
                cudaMemcpyHostToDevice);
            cudaMemcpy(
                (void*)d_b_cached->get_device_ptr(),
                b.data(),
                n * sizeof(float),
                cudaMemcpyHostToDevice);
            cudaMemcpy(
                (void*)d_x_cached->get_device_ptr(),
                x.data(),
                n * sizeof(float),
                cudaMemcpyHostToDevice);

            // Call GPU implementation
            result = solveGPU(
                n,
                nnz,
                reinterpret_cast<const int*>(
                    d_csrRowPtr_cached->get_device_ptr()),
                reinterpret_cast<const int*>(
                    d_csrColInd_cached->get_device_ptr()),
                reinterpret_cast<const float*>(
                    d_csrVal_cached->get_device_ptr()),
                reinterpret_cast<const float*>(d_b_cached->get_device_ptr()),
                reinterpret_cast<float*>(d_x_cached->get_device_ptr()),
                config);

            // Copy result back
            if (result.converged) {
                cudaMemcpy(
                    x.data(),
                    (void*)d_x_cached->get_device_ptr(),
                    n * sizeof(float),
                    cudaMemcpyDeviceToHost);
            }
        }
        catch (const std::exception& e) {
            result.converged = false;
            result.error_message =
                std::string("cuSOLVER Cholesky error: ") + e.what();
        }

        return result;
    }

    // Dense matrix GPU interface using cusolverDnSpotrf (Cholesky
    // factorization) A: column-major dense matrix on GPU [n*n], assumed
    // symmetric positive definite b: dense vector on GPU [n] x: dense vector on
    // GPU (output) [n]
    SolverResult solveDenseGPU(
        int n,
        const float* d_A,
        const float* d_b,
        float* d_x,
        const SolverConfig& config = SolverConfig{}) override
    {
        SolverResult result;
        auto start_time = std::chrono::high_resolution_clock::now();

        try {
            // Query workspace size
            int lwork = 0;
            cusolverStatus_t status = cusolverDnSpotrf_bufferSize(
                cusolverDnHandle,
                CUBLAS_FILL_MODE_LOWER,  // Use lower triangular part
                n,
                const_cast<float*>(d_A),
                n,  // leading dimension
                &lwork);

            if (status != CUSOLVER_STATUS_SUCCESS) {
                result.converged = false;
                result.error_message =
                    "Failed to query Cholesky workspace size: " +
                    std::to_string(status);
                return result;
            }

            // Reallocate cached buffers if size changed
            if (n != cached_dense_n || lwork != cached_lwork) {
                // Free old buffers
                if (d_work_cached)
                    cudaFree(d_work_cached);
                if (d_A_copy_cached)
                    cudaFree(d_A_copy_cached);

                // Allocate new buffers
                cudaMalloc(&d_work_cached, lwork * sizeof(float));
                cudaMalloc(&d_A_copy_cached, n * n * sizeof(float));

                cached_dense_n = n;
                cached_lwork = lwork;
            }

            // Allocate d_info once (fixed size)
            if (!d_info_cached) {
                cudaMalloc(&d_info_cached, sizeof(int));
            }

            // Copy A to temporary buffer (will be overwritten with L)
            cudaMemcpy(
                d_A_copy_cached,
                d_A,
                n * n * sizeof(float),
                cudaMemcpyDeviceToDevice);

            // Perform Cholesky factorization: A = L * L^T
            status = cusolverDnSpotrf(
                cusolverDnHandle,
                CUBLAS_FILL_MODE_LOWER,
                n,
                d_A_copy_cached,
                n,  // leading dimension
                d_work_cached,
                lwork,
                d_info_cached);

            if (status != CUSOLVER_STATUS_SUCCESS) {
                result.converged = false;
                result.error_message =
                    "cusolverDnSpotrf failed: " + std::to_string(status);
                return result;
            }

            // Copy b to x (will be overwritten with solution)
            cudaMemcpy(d_x, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice);

            // Solve L * L^T * x = b using the factorized matrix
            // This is done in two steps:
            // 1. Solve L * y = b for y
            // 2. Solve L^T * x = y for x
            status = cusolverDnSpotrs(
                cusolverDnHandle,
                CUBLAS_FILL_MODE_LOWER,
                n,
                1,                // nrhs (number of right-hand sides)
                d_A_copy_cached,  // Factorized matrix (L in lower triangle)
                n,                // lda
                d_x,              // On input: b, on output: x
                n,                // ldb
                d_info_cached);

            if (status != CUSOLVER_STATUS_SUCCESS) {
                result.converged = false;
                result.error_message =
                    "cusolverDnSpotrs failed: status=" + std::to_string(status);
                return result;
            }

            // No cleanup needed - buffers are cached and reused

            result.converged = true;
            result.iterations = 1;  // Direct solver
            result.final_residual = 0.0f;

            if (config.verbose) {
                std::cout << "Dense Cholesky solve completed successfully (n="
                          << n << ")" << std::endl;
            }
        }
        catch (const std::exception& e) {
            result.converged = false;
            result.error_message =
                std::string("Dense Cholesky error: ") + e.what();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.solve_time =
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);

        return result;
    }
};

// Factory registration
std::unique_ptr<LinearSolver> createCuSolverCholeskySolver()
{
    return std::make_unique<CuSolverCholeskySolver>();
}

}  // namespace Solver

RUZINO_NAMESPACE_CLOSE_SCOPE
