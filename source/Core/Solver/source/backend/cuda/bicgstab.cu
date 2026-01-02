#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <RHI/cuda.hpp>
#include <RHI/internal/cuda_extension.hpp>
#include <RZSolver/Solver.hpp>
#include <iostream>

RUZINO_NAMESPACE_OPEN_SCOPE
namespace Solver {

// 在 namespace 级别定义静态函数
namespace {
    SolverResult performCleanBiCGStabImpl(
        cublasHandle_t cublasHandle,
        cusparseHandle_t cusparseHandle,
        const SolverConfig& config,
        int n,
        cusparseSpMatDescr_t matA_desc,
        Ruzino::cuda::CUDALinearBufferHandle dBuffer,
        Ruzino::cuda::CUDALinearBufferHandle d_b,
        Ruzino::cuda::CUDALinearBufferHandle d_x,
        Ruzino::cuda::CUDALinearBufferHandle d_r,
        Ruzino::cuda::CUDALinearBufferHandle d_r0,
        Ruzino::cuda::CUDALinearBufferHandle d_p,
        Ruzino::cuda::CUDALinearBufferHandle d_v,
        Ruzino::cuda::CUDALinearBufferHandle d_s,
        Ruzino::cuda::CUDALinearBufferHandle d_t,
        cusparseDnVecDescr_t vecX_desc,
        cusparseDnVecDescr_t vecP_desc,
        cusparseDnVecDescr_t vecV_desc,
        cusparseDnVecDescr_t vecS_desc,
        cusparseDnVecDescr_t vecT_desc)
    {
        SolverResult result;
        const float one = 1.0f, zero = 0.0f, minus_one = -1.0f;

        // Compute ||b||
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

        // r = b - A*x
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
            vecT_desc,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            (void*)dBuffer->get_device_ptr());

        cublasSaxpy(
            cublasHandle,
            n,
            &minus_one,
            reinterpret_cast<float*>(d_t->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1);

        // r0 = r (standard choice)
        cublasScopy(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_r0->get_device_ptr()),
            1);

        // p = r
        cublasScopy(
            cublasHandle,
            n,
            reinterpret_cast<float*>(d_r->get_device_ptr()),
            1,
            reinterpret_cast<float*>(d_p->get_device_ptr()),
            1);

        float rho_old = 1.0f, alpha = 1.0f, omega = 1.0f;

        // BiCGSTAB iterations - clean implementation
        for (int iter = 0; iter < config.max_iterations; ++iter) {
            // rho = r0^T * r
            float rho;
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r0->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                &rho);  // Breakdown check - simpler condition
            if (abs(rho) < 1e-12f * b_norm * b_norm) {
                result.error_message = "BiCGSTAB breakdown: rho too small";
                break;
            }

            if (iter > 0) {
                float beta = (rho / rho_old) * (alpha / omega);

                // p = r + beta * (p - omega * v)
                float neg_omega = -omega;
                cublasSaxpy(
                    cublasHandle,
                    n,
                    &neg_omega,
                    reinterpret_cast<float*>(d_v->get_device_ptr()),
                    1,
                    reinterpret_cast<float*>(d_p->get_device_ptr()),
                    1);
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
                    reinterpret_cast<float*>(d_r->get_device_ptr()),
                    1,
                    reinterpret_cast<float*>(d_p->get_device_ptr()),
                    1);
            }

            // v = A * p
            cusparseSpMV(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA_desc,
                vecP_desc,
                &zero,
                vecV_desc,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                (void*)dBuffer->get_device_ptr());

            // alpha = rho / (r0^T * v)
            float r0_dot_v;
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r0->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_v->get_device_ptr()),
                1,
                &r0_dot_v);

            if (abs(r0_dot_v) < 1e-12f * b_norm * b_norm) {
                result.error_message = "BiCGSTAB breakdown: r0^T * v too small";
                break;
            }

            alpha = rho / r0_dot_v;

            // s = r - alpha * v
            cublasScopy(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_s->get_device_ptr()),
                1);
            float neg_alpha = -alpha;
            cublasSaxpy(
                cublasHandle,
                n,
                &neg_alpha,
                reinterpret_cast<float*>(d_v->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_s->get_device_ptr()),
                1);

            // Check early convergence
            float s_norm;
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_s->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_s->get_device_ptr()),
                1,
                &s_norm);
            s_norm = sqrt(s_norm);

            if (s_norm / b_norm < config.tolerance) {
                // x = x + alpha * p
                cublasSaxpy(
                    cublasHandle,
                    n,
                    &alpha,
                    reinterpret_cast<float*>(d_p->get_device_ptr()),
                    1,
                    reinterpret_cast<float*>(d_x->get_device_ptr()),
                    1);

                result.converged = true;
                result.iterations = iter + 1;
                result.final_residual = s_norm / b_norm;
                break;
            }

            // t = A * s
            cusparseSpMV(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA_desc,
                vecS_desc,
                &zero,
                vecT_desc,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                (void*)dBuffer->get_device_ptr());

            // omega = (t^T * s) / (t^T * t)
            float t_dot_s, t_dot_t;
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_t->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_s->get_device_ptr()),
                1,
                &t_dot_s);
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_t->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_t->get_device_ptr()),
                1,
                &t_dot_t);
            if (t_dot_t < 1e-12f * b_norm * b_norm) {
                // Instead of breaking down, try to recover using a different
                // approach Use the intermediate solution x = x + alpha * p
                cublasSaxpy(
                    cublasHandle,
                    n,
                    &alpha,
                    reinterpret_cast<float*>(d_p->get_device_ptr()),
                    1,
                    reinterpret_cast<float*>(d_x->get_device_ptr()),
                    1);

                // Check if this gives acceptable solution
                float s_relative_residual = s_norm / b_norm;
                if (s_relative_residual < config.tolerance * 10) {
                    result.converged = true;
                    result.iterations = iter + 1;
                    result.final_residual = s_relative_residual;
                    break;
                }
                else {
                    result.error_message =
                        "BiCGSTAB breakdown: t^T * t too small";
                    break;
                }
            }

            omega = t_dot_s / t_dot_t;

            // x = x + alpha * p + omega * s
            cublasSaxpy(
                cublasHandle,
                n,
                &alpha,
                reinterpret_cast<float*>(d_p->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_x->get_device_ptr()),
                1);
            cublasSaxpy(
                cublasHandle,
                n,
                &omega,
                reinterpret_cast<float*>(d_s->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_x->get_device_ptr()),
                1);

            // r = s - omega * t
            cublasScopy(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_s->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1);
            float neg_omega2 = -omega;
            cublasSaxpy(
                cublasHandle,
                n,
                &neg_omega2,
                reinterpret_cast<float*>(d_t->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1);

            // Check convergence
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
                result.iterations = iter + 1;
                result.final_residual = relative_residual;
                if (config.verbose) {
                    std::cout << "BiCGSTAB converged in " << iter + 1
                              << " iterations, residual: " << relative_residual
                              << std::endl;
                }
                break;
            }

            if (abs(omega) < 1e-12f) {
                result.error_message = "BiCGSTAB breakdown: omega too small";
                break;
            }

            rho_old = rho;
            result.iterations = iter + 1;
        }

        if (!result.converged && result.error_message.empty()) {
            result.error_message = "Maximum iterations reached";
            // Compute final residual
            float r_norm;
            cublasSdot(
                cublasHandle,
                n,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                reinterpret_cast<float*>(d_r->get_device_ptr()),
                1,
                &r_norm);
            result.final_residual = sqrt(r_norm) / b_norm;
        }

        return result;
    }
}  // namespace

class CudaBiCGStabSolver : public LinearSolver {
   private:
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    bool initialized = false;

   public:
    CudaBiCGStabSolver()
    {
        if (cusparseCreate(&cusparseHandle) != CUSPARSE_STATUS_SUCCESS ||
            cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to initialize CUDA libraries");
        }
        initialized = true;
    }

    ~CudaBiCGStabSolver()
    {
        if (initialized) {
            cusparseDestroy(cusparseHandle);
            cublasDestroy(cublasHandle);
        }
    }

    std::string getName() const override
    {
        return "CUDA BiCGSTAB";
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
                std::cout << "CUDA BiCGSTAB: n=" << n << ", nnz=" << nnz
                          << std::endl;
            }

            // Convert to CSR format
            std::vector<int> csrRowPtr(n + 1, 0);
            std::vector<int> csrColInd(nnz);
            std::vector<float> csrValues(nnz);

            // CSR conversion
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
                Ruzino::cuda::create_cuda_linear_buffer(csrValues);
            auto d_csrRowPtr =
                Ruzino::cuda::create_cuda_linear_buffer(csrRowPtr);
            auto d_csrColInd =
                Ruzino::cuda::create_cuda_linear_buffer(csrColInd);
            auto d_b = Ruzino::cuda::create_cuda_linear_buffer<float>(n);
            auto d_x = Ruzino::cuda::create_cuda_linear_buffer<float>(n);
            auto d_r = Ruzino::cuda::create_cuda_linear_buffer<float>(n);
            auto d_r0 = Ruzino::cuda::create_cuda_linear_buffer<float>(n);
            auto d_p = Ruzino::cuda::create_cuda_linear_buffer<float>(n);
            auto d_v = Ruzino::cuda::create_cuda_linear_buffer<float>(n);
            auto d_s = Ruzino::cuda::create_cuda_linear_buffer<float>(n);
            auto d_t = Ruzino::cuda::create_cuda_linear_buffer<float>(n);

            // Copy input data
            d_b->assign_host_vector(
                std::vector<float>(b.data(), b.data() + b.size()));
            d_x->assign_host_vector(
                std::vector<float>(x.data(), x.data() + x.size()));

            // Create matrix descriptor
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

            // Create vector descriptors
            cusparseDnVecDescr_t vecX_desc, vecP_desc, vecV_desc, vecS_desc,
                vecT_desc;
            cusparseCreateDnVec(
                &vecX_desc,
                n,
                reinterpret_cast<void*>(d_x->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecP_desc,
                n,
                reinterpret_cast<void*>(d_p->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecV_desc,
                n,
                reinterpret_cast<void*>(d_v->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecS_desc,
                n,
                reinterpret_cast<void*>(d_s->get_device_ptr()),
                CUDA_R_32F);
            cusparseCreateDnVec(
                &vecT_desc,
                n,
                reinterpret_cast<void*>(d_t->get_device_ptr()),
                CUDA_R_32F);

            // SpMV workspace
            size_t bufferSize = 0;
            const float one = 1.0f, zero = 0.0f;
            cusparseSpMV_bufferSize(
                cusparseHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one,
                matA_desc,
                vecP_desc,
                &zero,
                vecV_desc,
                CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT,
                &bufferSize);
            auto dBuffer =
                Ruzino::cuda::create_cuda_linear_buffer<uint8_t>(bufferSize);

            auto iteration_start_time =
                std::chrono::high_resolution_clock::now();

            // Clean BiCGSTAB implementation
            result = performCleanBiCGStab(
                config,
                n,
                matA_desc,
                dBuffer,
                d_b,
                d_x,
                d_r,
                d_r0,
                d_p,
                d_v,
                d_s,
                d_t,
                vecX_desc,
                vecP_desc,
                vecV_desc,
                vecS_desc,
                vecT_desc);

            auto iteration_end_time = std::chrono::high_resolution_clock::now();
            result.solve_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    iteration_end_time - iteration_start_time);

            // Copy result back
            auto result_vec = d_x->get_host_vector<float>();
            x = Eigen::Map<Eigen::VectorXf>(
                result_vec.data(), result_vec.size());

            // Cleanup
            cusparseDestroySpMat(matA_desc);
            cusparseDestroyDnVec(vecX_desc);
            cusparseDestroyDnVec(vecP_desc);
            cusparseDestroyDnVec(vecV_desc);
            cusparseDestroyDnVec(vecS_desc);
            cusparseDestroyDnVec(vecT_desc);
        }
        catch (const std::exception& e) {
            result.error_message = e.what();
            result.converged = false;
        }

        return result;
    }

   private:
    SolverResult performCleanBiCGStab(
        const SolverConfig& config,
        int n,
        cusparseSpMatDescr_t matA_desc,
        Ruzino::cuda::CUDALinearBufferHandle dBuffer,
        Ruzino::cuda::CUDALinearBufferHandle d_b,
        Ruzino::cuda::CUDALinearBufferHandle d_x,
        Ruzino::cuda::CUDALinearBufferHandle d_r,
        Ruzino::cuda::CUDALinearBufferHandle d_r0,
        Ruzino::cuda::CUDALinearBufferHandle d_p,
        Ruzino::cuda::CUDALinearBufferHandle d_v,
        Ruzino::cuda::CUDALinearBufferHandle d_s,
        Ruzino::cuda::CUDALinearBufferHandle d_t,
        cusparseDnVecDescr_t vecX_desc,
        cusparseDnVecDescr_t vecP_desc,
        cusparseDnVecDescr_t vecV_desc,
        cusparseDnVecDescr_t vecS_desc,
        cusparseDnVecDescr_t vecT_desc)
    {
        // 委托给静态函数
        return performCleanBiCGStabImpl(
            cublasHandle,
            cusparseHandle,
            config,
            n,
            matA_desc,
            dBuffer,
            d_b,
            d_x,
            d_r,
            d_r0,
            d_p,
            d_v,
            d_s,
            d_t,
            vecX_desc,
            vecP_desc,
            vecV_desc,
            vecS_desc,
            vecT_desc);
    }
};

// Factory registration
std::unique_ptr<LinearSolver> createCudaBiCGStabSolver()
{
    return std::make_unique<CudaBiCGStabSolver>();
}

}  // namespace Solver

RUZINO_NAMESPACE_CLOSE_SCOPE
