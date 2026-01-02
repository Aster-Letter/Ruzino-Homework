#if RUZINO_WITH_CUDA

#include <RHI/internal/cuda_extension.hpp>
#include <RHI/internal/cuda_graph.hpp>
#include <iostream>
#include <vector>

using namespace Ruzino::cuda;

__global__ void scale_kernel(float* data, float scale, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] *= scale;
    }
}

__global__ void add_kernel(float* a, float* b, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        a[tid] += b[tid];
    }
}

void cuda_graph_example()
{
    const int N = 1024;
    auto d_data = create_cuda_linear_buffer<float>(1.0f, N);  // 初始化为1.0f
    auto d_temp = create_cuda_linear_buffer<float>(1.0f, N);  // 初始化为1.0f

    cudaStream_t stream;
    cudaStreamCreate(
        &stream);  // Pattern 1: Simple lambda capture - Most elegant!
    std::cout << "=== Pattern 1: Lambda Capture ===" << std::endl;
    auto my_graph = with_cuda_graph(stream, [=]() {
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        scale_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_data->get_device_ptr<float>(), 2.0f, N);
        add_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_data->get_device_ptr<float>(),
            d_temp->get_device_ptr<float>(),
            N);
    });
    my_graph->launch(stream);
    cudaStreamSynchronize(stream);
    auto result1 = d_data->get_host_vector<float>();
    for (int i = 0; i < N; ++i)
        assert(result1[i] == 3.0f);  // 2.0f * (初始值应该是1.0f) + 1.0f = 3.0f
    std::cout << "Lambda captured graph launched! Result check passed."
              << std::endl;

    // Pattern 2: RAII with explicit operations
    std::cout << "\n=== Pattern 2: RAII with Explicit Operations ==="
              << std::endl;
    {
        auto graph = create_cuda_graph(stream);
        CUDAGraphCapture capture(graph);
        if (capture) {
            int blockSize = 128;
            int gridSize = (N + blockSize - 1) / blockSize;
            scale_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_data->get_device_ptr<float>(), 0.5f, N);
            scale_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_data->get_device_ptr<float>(), 3.0f, N);
        }
        CUDAGraphExecHandle exec = capture;
        exec->launch(stream);
    }
    cudaStreamSynchronize(stream);
    auto result2 = d_data->get_host_vector<float>();
    for (int i = 0; i < N; ++i)
        assert(result2[i] == (2.0f + 1.0f) * 0.5f * 3.0f);
    std::cout << "RAII captured graph launched!" << std::endl;

    // Pattern 3: 直接操作符使用
    std::cout << "\n=== Pattern 3: Direct Operator Usage ===" << std::endl;
    auto another_graph = capture_graph(stream, [=]() {
        int blockSize = 64;
        int gridSize = (N + blockSize - 1) / blockSize;
        scale_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_data->get_device_ptr<float>(), 1.5f, N);
    });
    another_graph->launch(stream);
    cudaStreamSynchronize(stream);
    auto result3 = d_data->get_host_vector<float>();
    for (int i = 0; i < N; ++i)
        assert(result3[i] == (2.0f + 1.0f) * 0.5f * 3.0f * 1.5f);
    std::cout << "Direct operator graph launched!" << std::endl;

    // Pattern 4: 移动语义
    std::cout << "\n=== Pattern 4: Move Semantics ===" << std::endl;
    auto create_complex_graph = [=]() -> CUDAGraphCapture {
        return with_cuda_graph(stream, [=]() {
            int blockSize = 512;
            int gridSize = (N + blockSize - 1) / blockSize;
            for (int i = 0; i < 5; ++i) {
                scale_kernel<<<gridSize, blockSize, 0, stream>>>(
                    d_data->get_device_ptr<float>(), 1.1f, N);
            }
        });
    };
    auto complex_graph = create_complex_graph();
    complex_graph->launch(stream);
    cudaStreamSynchronize(stream);
    auto result4 = d_data->get_host_vector<float>();
    float expected = (2.0f + 1.0f) * 0.5f * 3.0f * 1.5f;
    for (int i = 0; i < 5; ++i)
        expected *= 1.1f;
    for (int i = 0; i < N; ++i)
        assert(result4[i] == expected);
    std::cout << "Move semantics graph launched!" << std::endl;

    // Pattern 5: 释放所有权
    std::cout << "\n=== Pattern 5: Release Ownership ===" << std::endl;
    auto temp_capture = with_cuda_graph(stream, [=]() {
        scale_kernel<<<256, 64, 0, stream>>>(
            d_data->get_device_ptr<float>(), 2.5f, N);
    });
    CUDAGraphExecHandle released = temp_capture.release();
    released->launch(stream);
    cudaStreamSynchronize(stream);
    auto result5 = d_data->get_host_vector<float>();
    for (int i = 0; i < N; ++i)
        assert(result5[i] == expected * 2.5f);
    std::cout << "Released ownership graph launched!" << std::endl;

    cudaStreamDestroy(stream);
}

int main()
{
    std::cout << "Elegant CUDA Graph RAII Patterns" << std::endl;
    std::cout << "=================================" << std::endl;

    // Initialize CUDA
    if (cuda_init() != 0) {
        std::cerr << "Failed to initialize CUDA!" << std::endl;
        return -1;
    }

    try {
        cuda_graph_example();
        std::cout << "\nAll patterns demonstrated successfully!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    cuda_shutdown();
    return 0;
}

#else

#include <iostream>

int main()
{
    std::cout << "CUDA support not available!" << std::endl;
    return 0;
}

#endif  // RUZINO_WITH_CUDA
