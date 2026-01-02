#include <Eigen/Eigen>
#include <RHI/cuda.hpp>
#include <RHI/rhi.hpp>

#include "RHI/internal/cuda_extension.hpp"


void foo()
{
    Ruzino::cuda::GPUParallelFor(
        "Test Compiling", 10, GPU_LAMBDA_Ex(int idx) {
            Eigen::Vector3f v(1.0f, 2.0f, 3.0f);
            v = v.normalized();
        });
}