#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <RHI/cuda.hpp>
#include <RHI/internal/cuda_extension.hpp>
#include <RZSolver/Solver.hpp>
#include <cmath>
#include <iostream>

USTC_CG_NAMESPACE_OPEN_SCOPE

namespace Solver {

// 在 namespace 级别定义静态函数 - 回到基础的、可靠的实现
namespace {
    SolverResult performOptimizedGMRESImpl(
        cublasHandle_t cublasHandle,
        cusparseHandle_t cusparseHandle,
        const SolverConfig& config,
        int n,
        int restart,
        cusparseSpMatDescr_t matA_desc,
        USTC_CG::cuda::CUDALinearBufferHandle dBuffer,
        USTC_CG::cuda::CUDALinearBufferHandle d_b,
        USTC_CG::cuda::CUDALinearBufferHandle d_x,
        USTC_CG::cuda::CUDALinearBufferHandle d_r,
        USTC_CG::cuda::CUDALinearBufferHandle d_w,
        USTC_CG::cuda::CUDALinearBufferHandle d_V,
        cusparseDnVecDescr_t vecX_desc,
        cusparseDnVecDescr_t vecW_desc)
    {
        SolverResult result;
        const float one = 1.0f, zero = 0.0f, minus_one = -1.0f;

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

        if (b_norm == 0.0f) {
            result.converged = true;
            result.iterations = 0;
            result.final_residual = 0.0f;
            return result;
        }

        int total_iterations = 0;
        int max_restarts = std::max(1, config.max_iterations / restart);

        for (int restart_count = 0; restart_count < max_restarts;
             ++restart_count) {
            // Compute residual r = b - A*x
            cublasScopy(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_b->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1);

            cusparseSpMV(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA_desc,
                vecX_desc,
                &zero,
                vecW_desc,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                (void*)dBuffer->get_device_ptr());

            cublasSaxpy(
                cublasHandle,
                n,
                &minus_one,
                reinterpret_cast<float*>(d_w->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1);

            float r_norm;
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                &r_norm);
            r_norm = sqrt(r_norm);

            float relative_residual = r_norm / b_norm;
            if (relative_residual < config.tolerance) {
                result.converged = true;
                result.iterations = total_iterations;
                result.final_residual = relative_residual;
                return result;
            }

            // v1 = r / ||r||
            float inv_r_norm = 1.0f / r_norm;
            cublasSscal(
                cublasHandle,
                n,
                &inv_r_norm,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1);
            cublasScopy(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_V->get_device_ptr()),
                1);

            // 简化的HOST端GMRES - 移除所有GPU kernel开销
            std::vector<float> H((restart + 1) * restart, 0.0f);
            std::vector<float> g(restart + 1, 0.0f);
            g[0] = r_norm;
            std::vector<float> cs(restart, 0.0f);
            std::vector<float> sn(restart, 0.0f);

            int m;
            for (m = 0;
                 m < restart && total_iterations + m < config.max_iterations;
                 ++m) {
                // w = A * v_m
                cusparseDnVecDescr_t vecVm_desc;
                cusparseCreateDnVec(
                    &vecVm_desc,
                    n,
                    reinterpret_cast<void*>(
                        reinterpret_cast<float*>(d_V->get_device_ptr()) +
                        m * n),
                    CUDA_R_32F);

                cusparseSpMV(
                    cusparseHandle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &one,
                    matA_desc,
                    vecVm_desc,
                    &zero,
                    vecW_desc,
                    CUDA_R_32F,
                    CUSPARSE_SPMV_ALG_DEFAULT,
                    (void*)dBuffer->get_device_ptr());

                // Classical Gram-Schmidt - 只在这里优化
                for (int i = 0; i <= m; ++i) {
                    float hij;
                    cublasSdot(
                        cublasHandle,
                        n,
                        reinterpret_cast<float*>(d_w->get_device_ptr()),
                        1,
                        reinterpret_cast<float*>(d_V->get_device_ptr()) + i * n,
                        1,
                        &hij);

                    H[i + m * (restart + 1)] = hij;

                    float neg_hij = -hij;
                    cublasSaxpy(
                        cublasHandle,
                        n,
                        &neg_hij,
                        reinterpret_cast<float*>(d_V->get_device_ptr()) + i * n,
                        1,
                        reinterpret_cast<float*>(d_w->get_device_ptr()),
                        1);
                }

                // h_{m+1,m} = ||w||
                float hmm1;
                cublasSdot(
                    cublasHandle,
                    n,
                    reinterpret_cast<float*>(d_w->get_device_ptr()),
                    1,
                    reinterpret_cast<float*>(d_w->get_device_ptr()),
                    1,
                    &hmm1);
                hmm1 = sqrt(hmm1);
                H[(m + 1) + m * (restart + 1)] = hmm1;

                if (hmm1 < 1e-12f)
                    break;

                // v_{m+1} = w / h_{m+1,m}
                if (m + 1 < restart) {
                    float inv_hmm1 = 1.0f / hmm1;
                    cublasSscal(
                        cublasHandle,
                        n,
                        &inv_hmm1,
                        reinterpret_cast<float*>(d_w->get_device_ptr()),
                        1);
                    cublasScopy(
                        cublasHandle,
                        n,
                        reinterpret_cast<float*>(d_w->get_device_ptr()),
                        1,
                        reinterpret_cast<float*>(d_V->get_device_ptr()) +
                            (m + 1) * n,
                        1);
                }

                // Apply previous Givens rotations (HOST - 最快的方式)
                for (int i = 0; i < m; ++i) {
                    float h_im = H[i + m * (restart + 1)];
                    float h_i1m = H[(i + 1) + m * (restart + 1)];
                    H[i + m * (restart + 1)] = cs[i] * h_im + sn[i] * h_i1m;
                    H[(i + 1) + m * (restart + 1)] =
                        -sn[i] * h_im + cs[i] * h_i1m;
                }

                // Generate new Givens rotation (HOST)
                float h_mm = H[m + m * (restart + 1)];
                float h_m1m = H[(m + 1) + m * (restart + 1)];

                float norm = sqrt(h_mm * h_mm + h_m1m * h_m1m);
                if (norm > 1e-14f) {
                    cs[m] = h_mm / norm;
                    sn[m] = h_m1m / norm;
                }
                else {
                    cs[m] = 1.0f;
                    sn[m] = 0.0f;
                }

                // Apply to H and g
                H[m + m * (restart + 1)] = cs[m] * h_mm + sn[m] * h_m1m;
                H[(m + 1) + m * (restart + 1)] = 0.0f;

                float g_m = g[m];
                g[m] = cs[m] * g_m;
                g[m + 1] = -sn[m] * g_m;

                // Check convergence
                relative_residual = abs(g[m + 1]) / b_norm;
                if (relative_residual < config.tolerance) {
                    m++;
                    break;
                }

                cusparseDestroyDnVec(vecVm_desc);
            }

            // Back substitution (HOST)
            std::vector<float> y(m, 0.0f);
            for (int i = m - 1; i >= 0; --i) {
                y[i] = g[i];
                for (int j = i + 1; j < m; ++j) {
                    y[i] -= H[i + j * (restart + 1)] * y[j];
                }
                float h_ii = H[i + i * (restart + 1)];
                if (abs(h_ii) < 1e-15f) {
                    result.error_message = "GMRES: Singular Hessenberg matrix";
                    result.converged = false;
                    return result;
                }
                y[i] /= h_ii;
            }

            // Update solution: x = x + V * y (批量操作以减少开销)
            for (int i = 0; i < m; ++i) {
                cublasSaxpy(
                    cublasHandle,
                    n,
                    &y[i],
                    reinterpret_cast<float*>(d_V->get_device_ptr()) + i * n,
                    1,
                    reinterpret_cast<float*>(d_x->get_device_ptr()),
                    1);
            }

            total_iterations += m;
            result.iterations = total_iterations;

            if (relative_residual < config.tolerance) {
                result.converged = true;
                result.final_residual = relative_residual;
                if (config.verbose) {
                    std::cout << "CUDA GMRES converged in " << total_iterations
                              << " iterations, residual: " << relative_residual
                              << std::endl;
                }
                break;
            }

            if (total_iterations >= config.max_iterations)
                break;
        }

        if (!result.converged) {
            result.error_message = "Maximum iterations reached";
            // Compute final residual
            cublasScopy(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_b->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1);

            cusparseSpMV(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA_desc,
                vecX_desc,
                &zero,
                vecW_desc,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                (void*)dBuffer->get_device_ptr());

            cublasSaxpy(
                cublasHandle,
                n,
                &minus_one,
                reinterpret_cast<float*>(d_w->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1);

            float final_r_norm;
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                &final_r_norm);

            result.final_residual = sqrt(final_r_norm) / b_norm;
        }

        return result;
    }
}  // namespace

class CudaGMRESSolver : public LinearSolver {
   private:
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    bool initialized = false;

   public:
    CudaGMRESSolver()
    {
        if (cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS ||
            cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to initialize CUDA libraries");
        }
        initialized = true;
    }

    ~CudaGMRESSolver()
    {
        if (initialized) {
            cusparseDestroy(cusparseHandle);
            cublasDestroy(cublasHandle);
        }
    }

    std::string getName() const override
    {
        return "CUDA GMRES";
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
            // 进一步减小重启大小以提高效率
            int restart =
                std::min(10, std::max(3, n / 200));  // 更激进的重启大小

            if (config.verbose) {
                std::cout << "CUDA GMRES: n=" << n << ", nnz=" << nnz
                          << ", restart=" << restart << std::endl;
            }

            // CSR conversion (same as BiCGSTAB)
            std::vector<int> csrRowPtr(n + 1, 0);
            std::vector<int> csrColInd(nnz);
            std::vector<float> csrValues(nnz);

            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it;
                     ++it) {
                    csrRowPtr[it.row() + 1]++;
                }
            }
            for (int i = 1; i <= n; ++i) {
                csrRowPtr[i] += csrRowPtr[i - 1];
            }
            std::vector<int> current_pos = csrRowPtr;
            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<float>::InnerIterator it(A, k); it;
                     ++it) {
                    int row = it.row();
                    int pos = current_pos[row]++;
                    csrValues[pos] = it.value();
                    csrColInd[pos] = it.col();
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
            auto d_b = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
            auto d_x = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
            auto d_r = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
            auto d_w = USTC_CG::cuda::create_cuda_linear_buffer<float>(n);
            auto d_V = USTC_CG::cuda::create_cuda_linear_buffer<float>(
                n * (restart + 1));

            d_b->assign_host_vector(
                std::vector<float>(b.data(), b.data() + b.size()));
            d_x->assign_host_vector(
                std::vector<float>(x.data(), x.data() + x.size()));

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

            cusparseDnVecDescr_t vecX_desc, vecW_desc;
            cusparseCreateDnVec(
                &vecX_desc,
                n,
                reinterpret_cast<void*>(d_x->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecW_desc,
                n,
                reinterpret_cast<void*>(d_w->get_device_ptr()),
                CUDA_R_32F);

            size_t bufferSize = 0;
            const float one = 1.0f, zero = 0.0f;
            cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA_desc,
                vecX_desc,
                &zero,
                vecW_desc,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSize);
            auto dBuffer =
                USTC_CG::cuda::create_cuda_linear_buffer<uint8_t>(bufferSize);

            auto iteration_start_time =
                std::chrono::high_resolution_clock::now();

            // 简化的GMRES实现 - 专注于正确性和效率
            result = performOptimizedGMRES(
                config,
                n,
                restart,
                matA_desc,
                dBuffer,
                d_b,
                d_x,
                d_r,
                d_w,
                d_V,
                vecX_desc,
                vecW_desc);

            auto iteration_end_time = std::chrono::high_resolution_clock::now();
            result.solve_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    iteration_end_time - iteration_start_time);

            auto result_vec = d_x->get_host_vector<float>();
            x = Eigen::Map<Eigen::VectorXf>(
                result_vec.data(), result_vec.size());

            cusparseDestroySpMat(matA_desc);
            cusparseDestroyDnVec(vecX_desc);
            cusparseDestroyDnVec(vecW_desc);
        }
        catch (const std::exception& e) {
            result.error_message = e.what();
            result.converged = false;
        }

        return result;
    }

   private:
    SolverResult performOptimizedGMRES(
        const SolverConfig& config,
        int n,
        int restart,
        cusparseSpMatDescr_t matA_desc,
        USTC_CG::cuda::CUDALinearBufferHandle dBuffer,
        USTC_CG::cuda::CUDALinearBufferHandle d_b,
        USTC_CG::cuda::CUDALinearBufferHandle d_x,
        USTC_CG::cuda::CUDALinearBufferHandle d_r,
        USTC_CG::cuda::CUDALinearBufferHandle d_w,
        USTC_CG::cuda::CUDALinearBufferHandle d_V,
        cusparseDnVecDescr_t vecX_desc,
        cusparseDnVecDescr_t vecW_desc)
    {
        return performOptimizedGMRESImpl(
            cublasHandle,
            cusparseHandle,
            config,
            n,
            restart,
            matA_desc,
            dBuffer,
            d_b,
            d_x,
            d_r,
            d_w,
            d_V,
            vecX_desc,
            vecW_desc);
    }
};

// Factory registration
std::unique_ptr<LinearSolver> createCudaGMRESSolver()
{
    return std::make_unique<CudaGMRESSolver>();
}

}  // namespace Solver
USTC_CG_NAMESPACE_CLOSE_SCOPE
