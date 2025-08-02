#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <RHI/cuda.hpp>
#include <RHI/internal/cuda_extension.hpp>
#include <RZSolver/Solver.hpp>
#include <iostream>

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace Solver {

class CudaCGSolver : public LinearSolver {
   private:
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    bool initialized = false;

   public:
    CudaCGSolver()
    {
        if (cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS ||
            cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to initialize CUDA libraries");
        }
        initialized = true;
    }

    ~CudaCGSolver()
    {
        if (initialized) {
            cusparseDestroy(cusparseHandle);
            cublasDestroy(cublasHandle);
        }
    }

    std::string getName() const override
    {
        return "CUDA Conjugate Gradient";
    }

    bool isIterative() const override
    {
        return true;
    }
    bool requiresGPU() const override
    {
        return true;
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

            if (config.verbose) {
                std::cout << "CUDA CG: n=" << n << ", nnz=" << nnz << std::endl;
            }

            // Convert to CSR format
            std::vector<int> csrRowPtr(n + 1, 0);
            std::vector<int> csrColInd(nnz);
            std::vector<float> csrValues(nnz);
            std::vector<float> diagonal(n, 1.0f);

            // First pass: count entries per row
            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it;
                     ++it) {
                    csrRowPtr[it.row() + 1]++;
                }
            }

            // Convert counts to offsets
            for (int i = 1; i <= n; ++i) {
                csrRowPtr[i] += csrRowPtr[i - 1];
            }

            // Second pass: fill values and column indices
            std::vector<int> current_pos = csrRowPtr;
            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it;
                     ++it) {
                    int row = it.row();
                    int pos = current_pos[row]++;
                    csrValues[pos] = it.value();
                    csrColInd[pos] = it.col();

                    if (it.row() == it.col()) {
                        diagonal[row] = it.value();
                    }
                }
            }

            auto setup_end_time = std::chrono::high_resolution_clock::now();
            result.setup_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    setup_end_time - start_time);

            // GPU setup
            auto d_csrValues =
                USTC_CG::cuda::create_cuda_linear_buffer(csrValues);
            auto d_csrRowPtr =
                USTC_CG::cuda::create_cuda_linear_buffer(csrRowPtr);
            auto d_csrColInd =
                USTC_CG::cuda::create_cuda_linear_buffer(csrColInd);
            auto d_diagonal =
                USTC_CG::cuda::create_cuda_linear_buffer(diagonal);
            auto d_b = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
            auto d_x = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
            auto d_r = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
            auto d_z = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
            auto d_p = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
            auto d_Ap = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);

            // Copy data to GPU
            d_b->assign_host_vector(
                std::vector<float>(b.data(), b.data() + b.size()));
            d_x->assign_host_vector(
                std::vector<float>(x.data(), x.data() + x.size()));

            // Create descriptors
            cusparseSpMatDescr_t matA_desc;
            cusparseCreateCsr(
                &matA_desc,
                n,
                n,
                nnz,
                reinterpret_cast<void*>(d_csrRowPtr->get_device_ptr()),
                reinterpret_cast<void*>(d_csrColInd->get_device_ptr()),
                reinterpret_cast<void*>(d_csrValues->get_device_ptr()),
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_32F);

            cusparseDnVecDescr_t vecX_desc, vecB_desc, vecR_desc, vecZ_desc,
                vecP_desc, vecAp_desc;
            cusparseCreateDnVec(
                &vecX_desc,
                n,
                reinterpret_cast<void*>(d_x->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecB_desc,
                n,
                reinterpret_cast<void*>(d_b->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecR_desc,
                n,
                reinterpret_cast<void*>(d_r->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecZ_desc,
                n,
                reinterpret_cast<void*>(d_z->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecP_desc,
                n,
                reinterpret_cast<void*>(d_p->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecAp_desc,
                n,
                reinterpret_cast<void*>(d_Ap->get_device_ptr()),
                CUDA_R_32F);

            // Allocate workspace
            size_t bufferSize = 0;
            const float one = 1.0f, zero = 0.0f, minus_one = -1.0f;
            cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA_desc,
                vecP_desc,
                &zero,
                vecAp_desc,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSize);
            auto dBuffer =
                USTC_CG::cuda::create_cuda_linear_buffer<uint8_t>(bufferSize);

            auto iteration_start_time =
                std::chrono::high_resolution_clock::now();

            // CG algorithm implementation
            result = performCGIterations(
                config,
                n,
                matA_desc,
                vecX_desc,
                vecB_desc,
                vecR_desc,
                vecZ_desc,
                vecP_desc,
                vecAp_desc,
                d_diagonal,
                dBuffer,
                d_b,
                d_x,
                d_r,
                d_z,
                d_p,
                d_Ap);

            // Copy result back
            auto result_vec = d_x->get_host_vector<float>();
            x = Eigen::Map<Eigen::VectorXf>(
                result_vec.data(), result_vec.size());

            auto iteration_end_time = std::chrono::high_resolution_clock::now();
            result.solve_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    iteration_end_time - iteration_start_time);

            // Cleanup
            cusparseDestroySpMat(matA_desc);
            cusparseDestroyDnVec(vecX_desc);
            cusparseDestroyDnVec(vecB_desc);
            cusparseDestroyDnVec(vecR_desc);
            cusparseDestroyDnVec(vecZ_desc);
            cusparseDestroyDnVec(vecP_desc);
            cusparseDestroyDnVec(vecAp_desc);
        }
        catch (const std::exception& e) {
            result.error_message = e.what();
            result.converged = false;
        }

        return result;
    }

   private:
    SolverResult performCGIterations(
        const SolverConfig& config,
        int n,
        cusparseSpMatDescr_t matA_desc,
        cusparseDnVecDescr_t vecX_desc,
        cusparseDnVecDescr_t vecB_desc,
        cusparseDnVecDescr_t vecR_desc,
        cusparseDnVecDescr_t vecZ_desc,
        cusparseDnVecDescr_t vecP_desc,
        cusparseDnVecDescr_t vecAp_desc,
        USTC_CG::cuda::CUDALinearBufferHandle d_diagonal,
        USTC_CG::cuda::CUDALinearBufferHandle dBuffer,
        USTC_CG::cuda::CUDALinearBufferHandle d_b,
        USTC_CG::cuda::CUDALinearBufferHandle d_x,
        USTC_CG::cuda::CUDALinearBufferHandle d_r,
        USTC_CG::cuda::CUDALinearBufferHandle d_z,
        USTC_CG::cuda::CUDALinearBufferHandle d_p,
        USTC_CG::cuda::CUDALinearBufferHandle d_Ap)
    {
        SolverResult result;
        const float one = 1.0f, zero = 0.0f, minus_one = -1.0f;

        // Compute ||b|| for relative residual
        float b_norm;
        cublasSdot(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_b->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_b->get_device_ptr()),
            1,
            &b_norm);
        b_norm = sqrt(b_norm);

        // r = b - A*x
        // First copy b to r
        cublasScopy(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_b->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1);

        // Then compute A*x and subtract from r
        cusparseSpMV(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            matA_desc,
            vecX_desc,
            &zero,
            vecAp_desc,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            (void*)dBuffer->get_device_ptr());

        cublasSaxpy(
            cublasHandle,
            n,
            &minus_one,
            reinterpret_cast<float*>(d_Ap->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1);

        // Apply preconditioner if enabled
        if (config.use_preconditioner) {
            // Simple diagonal preconditioning: z[i] = r[i] / diagonal[i]
            // Use element-wise division kernel or approximate with identity for now
            cublasScopy(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_z->get_device_ptr()),
                1);
        }
        else {
            cublasScopy(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_z->get_device_ptr()),
                1);
        }

        // p = z
        cublasScopy(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_z->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_p->get_device_ptr()),
            1);

        // rzold = r^T * z
        float rzold, rznew, alpha, beta;
        cublasSdot(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_z->get_device_ptr()),
            1,
            &rzold);

        float initial_residual = sqrt(rzold);
        if (config.verbose) {
            std::cout << "Initial residual norm: " << initial_residual
                      << std::endl;
        }

        // Check if already converged
        if (initial_residual / b_norm < config.tolerance) {
            result.converged = true;
            result.iterations = 0;
            result.final_residual = initial_residual / b_norm;
            return result;
        }

        // CG iterations
        for (int iter = 0; iter < config.max_iterations; ++iter) {
            // Ap = A * p
            cusparseSpMV(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA_desc,
                vecP_desc,
                &zero,
                vecAp_desc,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                reinterpret_cast<void*>(dBuffer->get_device_ptr()));

            // alpha = rzold / (p^T * Ap)
            float pAp;
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_p->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_Ap->get_device_ptr()),
                1,
                &pAp);

            if (abs(pAp) < 1e-20f) {
                result.error_message = "CG breakdown: p^T * A * p near zero";
                break;
            }

            alpha = rzold / pAp;

            // x = x + alpha * p
            cublasSaxpy(
                cublasHandle,
                n,
                &alpha,
                reinterpret_cast<float*>(d_p->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_x->get_device_ptr()),
                1);

            // r = r - alpha * Ap
            float neg_alpha = -alpha;
            cublasSaxpy(
                cublasHandle,
                n,
                &neg_alpha,
                reinterpret_cast<float*>(d_Ap->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1);

            // Apply preconditioner
            if (config.use_preconditioner) {
                cublasScopy(
                    cublasHandle,
                    n,
                    reinterpret_cast<float*>(d_r->get_device_ptr()),
                    1,
                    reinterpret_cast<float*>(d_z->get_device_ptr()),
                    1);
            }
            else {
                cublasScopy(
                    cublasHandle,
                    n,
                    reinterpret_cast<float*>(d_r->get_device_ptr()),
                    1,
                    reinterpret_cast<float*>(d_z->get_device_ptr()),
                    1);
            }

            // rznew = r^T * z
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_z->get_device_ptr()),
                1,
                &rznew);

            // Check convergence
            float relative_residual = sqrt(rznew) / b_norm;
            if (relative_residual < config.tolerance) {
                result.converged = true;
                result.iterations = iter + 1;
                result.final_residual = relative_residual;

                if (config.verbose) {
                    std::cout << "CUDA CG converged in " << iter + 1
                              << " iterations, residual: " << relative_residual
                              << std::endl;
                }
                break;
            }

            if (abs(rzold) < 1e-20f) {
                result.error_message = "CG breakdown: r^T * z near zero";
                break;
            }

            // beta = rznew / rzold
            beta = rznew / rzold;

            // p = z + beta * p
            cublasSscal(
                cublasHandle,
                n,
                &beta,
                reinterpret_cast<float*>(d_p->get_device_ptr()),
                1);
            cublasSaxpy(
                cublasHandle,
                n,
                &one,
                reinterpret_cast<float*>(d_z->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_p->get_device_ptr()),
                1);

            rzold = rznew;
            result.iterations = iter + 1;
        }

        if (!result.converged && result.error_message.empty()) {
            result.error_message =
                "Maximum iterations reached without convergence";
            result.final_residual = sqrt(rzold) / b_norm;
        }

        return result;
    }
};

// Factory registration
std::unique_ptr<LinearSolver> createCudaCGSolver()
{
    return std::make_unique<CudaCGSolver>();
}

}  // namespace Solver

USTC_CG_NAMESPACE_CLOSE_SCOPE
