#if RUZINO_WITH_CUDA

#include <RHI/internal/cuda_extension.hpp>
#include <RHI/internal/cuda_graph.hpp>
#include <iostream>
#include <memory>
#include <vector>

using namespace Ruzino::cuda;

__global__ void dynamic_kernel(float* data, float* temp, int n, float factor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = data[tid] * factor + temp[tid];
    }
}

__global__ void reduction_kernel(float* input, float* output, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(output, sdata[0]);
}

// Host function for graph node
void host_callback(void* userData)
{
    float* result = static_cast<float*>(userData);
    std::cout << "Host callback executed! Current result: " << *result
              << std::endl;
}

void advanced_cuda_graph_examples()
{
    const int N = 1024;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "=== Advanced CUDA Graph Examples ===" << std::endl;

    // Example 1: 使用显式图构建API (Manual Graph Construction)
    std::cout << "\n--- Example 1: Manual Graph Construction ---" << std::endl;
    {
        cudaGraph_t graph;
        cudaGraphCreate(&graph, 0);

        // 1. 添加内存分配节点
        cudaMemAllocNodeParams allocParams = {};
        allocParams.bytesize = N * sizeof(float);
        allocParams.poolProps.allocType = cudaMemAllocationTypePinned;
        allocParams.poolProps.location.type = cudaMemLocationTypeDevice;
        allocParams.poolProps.location.id = 0;

        cudaGraphNode_t allocNode;
        cudaGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParams);

        // 2. 添加内存设置节点
        cudaMemsetParams memsetParams = {};
        memsetParams.dst = allocParams.dptr;
        memsetParams.value = 0;
        memsetParams.pitch = 0;
        memsetParams.elementSize = sizeof(float);
        memsetParams.width = N;
        memsetParams.height = 1;

        cudaGraphNode_t memsetNode;
        cudaGraphAddMemsetNode(
            &memsetNode, graph, &allocNode, 1, &memsetParams);

        // 3. 添加核函数节点
        auto d_temp = create_cuda_linear_buffer<float>(2.0f, N);

        // Prepare kernel arguments
        auto dataPtr = allocParams.dptr;
        auto tempPtr = d_temp->get_device_ptr<float>();
        int nVal = N;
        float factorVal = 3.0f;
        void* kernelArgs[] = { &dataPtr, &tempPtr, &nVal, &factorVal };

        cudaKernelNodeParams kernelParams = {};
        kernelParams.func = (void*)dynamic_kernel;
        kernelParams.gridDim = dim3((N + 255) / 256);
        kernelParams.blockDim = dim3(256);
        kernelParams.kernelParams = kernelArgs;
        kernelParams.sharedMemBytes = 0;

        cudaGraphNode_t kernelNode;
        cudaGraphAddKernelNode(
            &kernelNode, graph, &memsetNode, 1, &kernelParams);

        // 4. 添加内存释放节点
        cudaGraphNode_t freeNode;
        cudaGraphAddMemFreeNode(
            &freeNode, graph, &kernelNode, 1, allocParams.dptr);

        // 实例化并执行
        cudaGraphExec_t graphExec;
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);

        std::cout << "Manual graph construction completed!" << std::endl;

        // 清理
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    }

    // Example 2: 条件图和分支 (Conditional Graphs)
    std::cout << "\n--- Example 2: Conditional Execution ---" << std::endl;
    {
        auto d_data = create_cuda_linear_buffer<float>(1.0f, N);
        auto d_temp = create_cuda_linear_buffer<float>(0.5f, N);
        auto d_result = create_cuda_linear_buffer<float>(0.0f, 1);

        // 创建条件图：根据条件选择不同的处理路径
        auto conditional_graph = with_cuda_graph(stream, [=]() {
            dim3 grid((N + 255) / 256);
            dim3 block(256);

            // Path A: 简单处理
            dynamic_kernel<<<grid, block, 0, stream>>>(
                d_data->get_device_ptr<float>(),
                d_temp->get_device_ptr<float>(),
                N,
                2.0f);

            // Path B: 复杂处理 (reduction)
            reduction_kernel<<<grid, block, block.x * sizeof(float), stream>>>(
                d_data->get_device_ptr<float>(),
                d_result->get_device_ptr<float>(),
                N);
        });

        conditional_graph->launch(stream);
        cudaStreamSynchronize(stream);
        auto final_result = d_result->get_host_value<float>();
        std::cout << "Conditional graph result: " << final_result << std::endl;
    }

    // 确保主流完全同步，避免影响后续示例
    cudaStreamSynchronize(stream);  // Example 3: 真正的并行流图执行 (True
                                    // Parallel Multi-stream Graph Execution)
    std::cout
        << "\n--- Example 3: True Parallel Multi-stream Graph Execution ---"
        << std::endl;
    {
        // 创建多个独立的流
        cudaStream_t stream1, stream2, stream3;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);

        auto d_data1 = create_cuda_linear_buffer<float>(1.0f, N);
        auto d_data2 = create_cuda_linear_buffer<float>(2.0f, N);
        auto d_data3 = create_cuda_linear_buffer<float>(3.0f, N);

        try {
            // 阶段1：创建第一个图
            std::cout << "Creating graph 1..." << std::endl;
            auto graph1 = with_cuda_graph(stream1, [=]() {
                int blockSize = 256;
                int gridSize = (N + blockSize - 1) / blockSize;
                dynamic_kernel<<<gridSize, blockSize, 0, stream1>>>(
                    d_data1->get_device_ptr<float>(),
                    d_data1->get_device_ptr<float>(),
                    N,
                    1.5f);
            });

            // 阶段2：启动第一个图，同时创建第二个图
            std::cout << "Launching graph 1 while creating graph 2..."
                      << std::endl;
            if (graph1) {
                graph1->launch(stream1);  // 异步启动
            }

            // 在graph1运行的同时创建graph2
            auto graph2 = with_cuda_graph(stream2, [=]() {
                int blockSize = 256;
                int gridSize = (N + blockSize - 1) / blockSize;
                dynamic_kernel<<<gridSize, blockSize, 0, stream2>>>(
                    d_data2->get_device_ptr<float>(),
                    d_data2->get_device_ptr<float>(),
                    N,
                    2.5f);
            });

            // 阶段3：启动第二个图，同时创建第三个图
            std::cout << "Launching graph 2 while creating graph 3..."
                      << std::endl;
            if (graph2) {
                graph2->launch(stream2);  // 异步启动
            }

            // 在graph2运行的同时创建graph3
            auto graph3 = with_cuda_graph(stream3, [=]() {
                int blockSize = 256;
                int gridSize = (N + blockSize - 1) / blockSize;
                dynamic_kernel<<<gridSize, blockSize, 0, stream3>>>(
                    d_data3->get_device_ptr<float>(),
                    d_data3->get_device_ptr<float>(),
                    N,
                    3.5f);
            });

            // 阶段4：启动第三个图
            std::cout << "Launching graph 3..." << std::endl;
            if (graph3) {
                graph3->launch(stream3);
            }

            // 等待所有图执行完成
            std::cout << "Waiting for all graphs to complete..." << std::endl;
            cudaStreamSynchronize(stream1);
            cudaStreamSynchronize(stream2);
            cudaStreamSynchronize(stream3);

            std::cout << "All three graphs executed in overlapping fashion!"
                      << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error in parallel execution example: " << e.what()
                      << std::endl;
        }

        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);
    }

    // Example 4: 图更新和参数修改 (Graph Updates)
    std::cout << "\n--- Example 4: Dynamic Graph Updates ---" << std::endl;
    {
        auto d_data = create_cuda_linear_buffer<float>(1.0f, N);
        auto d_temp = create_cuda_linear_buffer<float>(1.0f, N);

        // 创建基础图
        auto base_graph = capture_graph(stream, [=]() {
            dim3 grid((N + 255) / 256);
            dim3 block(256);
            dynamic_kernel<<<grid, block, 0, stream>>>(
                d_data->get_device_ptr<float>(),
                d_temp->get_device_ptr<float>(),
                N,
                2.0f);
        });

        // 第一次执行
        base_graph->launch(stream);
        cudaStreamSynchronize(stream);
        auto result_v1 = d_data->get_host_vector<float>();
        std::cout << "Graph v1 result: " << result_v1[0] << std::endl;

        // 创建更新的图 (修改参数)
        auto updated_graph = capture_graph(stream, [=]() {
            dim3 grid((N + 255) / 256);
            dim3 block(256);
            dynamic_kernel<<<grid, block, 0, stream>>>(
                d_data->get_device_ptr<float>(),
                d_temp->get_device_ptr<float>(),
                N,
                5.0f);  // 不同的参数
        });

        // 尝试更新图（如果拓扑兼容）
        if (base_graph.get()->update(create_cuda_graph(stream))) {
            std::cout << "Graph updated successfully!" << std::endl;
        }
        else {
            std::cout << "Graph update failed, using new graph" << std::endl;
            base_graph = std::move(updated_graph);
        }

        // 执行更新后的图
        base_graph->launch(stream);
        cudaStreamSynchronize(stream);
        auto result_v2 = d_data->get_host_vector<float>();
        std::cout << "Graph v2 result: " << result_v2[0] << std::endl;
    }  // Example 5: 使用高级图构建器和内存池 (Advanced Graph Builder with
       // Memory Pool)
    std::cout << "\n--- Example 5: Advanced Graph Builder with Memory Pool ---"
              << std::endl;
    {
        // 创建内存池
        CUDAGraphMemoryPool mempool(4 * 1024 * 1024);  // 4MB pool

        // 使用内存池创建高级图构建器
        auto builder = create_advanced_graph_with_pool(mempool.getPool());

        // 使用构建器创建复杂的图
        void* d_data_ptr = nullptr;
        void* d_temp_ptr = nullptr;
        void* d_result_ptr = nullptr;

        // 添加内存分配节点 (从内存池分配)
        builder.addMemAllocNode(&d_data_ptr, N * sizeof(float))
            .addMemAllocNode(&d_temp_ptr, N * sizeof(float))
            .addMemAllocNode(&d_result_ptr, sizeof(float));

        // 添加内存初始化节点
        auto allocNode1 = builder.getNode(0);
        auto allocNode2 = builder.getNode(1);
        auto allocNode3 = builder.getNode(2);

        // 初始化数据为1.0f, temp为2.0f, result为0.0f
        builder
            .addMemsetNode(
                d_data_ptr,
                0x3F800000,
                N * sizeof(float),
                { allocNode1 })  // 1.0f in hex
            .addMemsetNode(
                d_temp_ptr,
                0x40000000,
                N * sizeof(float),
                { allocNode2 })  // 2.0f in hex
            .addMemsetNode(d_result_ptr, 0, sizeof(float), { allocNode3 });

        // 添加Host回调节点
        float host_result = 42.0f;
        auto lastMemsetNode = builder.getLastNode();
        builder.addHostNode(host_callback, &host_result, { lastMemsetNode });

        // 添加内存释放节点 (回到内存池)
        auto hostNode = builder.getLastNode();
        builder.addMemFreeNode(d_data_ptr, { hostNode })
            .addMemFreeNode(d_temp_ptr, { hostNode })
            .addMemFreeNode(d_result_ptr, { hostNode });

        // 构建并执行图
        auto graph_exec = builder.build();
        if (graph_exec) {
            graph_exec->launch(stream);
            cudaStreamSynchronize(stream);
            std::cout << "Advanced graph builder with memory pool completed!"
                      << std::endl;
            std::cout << "Memory pool was used for allocation/deallocation!"
                      << std::endl;
        }
    }

    // Example 6: 纯 CUDAGraphBuilder 使用示例 (Pure CUDAGraphBuilder Usage)
    std::cout << "\n--- Example 6: Pure CUDAGraphBuilder Usage ---"
              << std::endl;
    {
        // 创建不带内存池的图构建器
        auto builder = create_advanced_graph();

        auto d_input = create_cuda_linear_buffer<float>(5.0f, N);
        auto d_output = create_cuda_linear_buffer<float>(0.0f, N);

        // 使用图构建器构建更复杂的图
        float scale_factor = 3.14f;
        float* factor_ptr = &scale_factor;

        // 添加Host节点作为开始
        builder.addHostNode(
            [](void* userData) {
                std::cout << "Graph execution started via builder!"
                          << std::endl;
            },
            nullptr);

        auto startNode = builder.getLastNode();

        // 添加内存设置节点来清零输出
        builder.addMemsetNode(
            reinterpret_cast<void*>(d_output->get_device_ptr<float>()),
            0,
            N * sizeof(float),
            { startNode });

        auto memsetNode = builder.getLastNode();

        // 添加Host回调节点
        builder.addHostNode(
            [](void* userData) {
                float* factor = static_cast<float*>(userData);
                std::cout << "Processing with scale factor: " << *factor
                          << std::endl;
            },
            factor_ptr,
            { memsetNode });

        // 构建并执行图
        auto graph_exec = builder.build();
        if (graph_exec) {
            graph_exec->launch(stream);
            cudaStreamSynchronize(stream);
            std::cout << "Pure CUDAGraphBuilder execution completed!"
                      << std::endl;
        }
    }
    // Example 7: 工作队列模式 (Work Queue Pattern) - 简化版本
    std::cout << "\n--- Example 7: Simplified Work Queue Pattern ---"
              << std::endl;
    {
        // 简化的工作队列实现
        auto d_input = create_cuda_linear_buffer<float>(1.0f, N);
        auto d_output = create_cuda_linear_buffer<float>(0.0f, N);

        auto queue_graph = with_cuda_graph(stream, [=]() {
            dim3 grid((N + 255) / 256);
            dim3 block(256);

            // 第一阶段：处理输入数据
            dynamic_kernel<<<grid, block, 0, stream>>>(
                d_input->get_device_ptr<float>(),
                d_input->get_device_ptr<float>(),
                N,
                2.0f);

            // 第二阶段：将结果复制到输出
            cudaMemcpyAsync(
                d_output->get_device_ptr<float>(),
                d_input->get_device_ptr<float>(),
                N * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream);
        });

        queue_graph->launch(stream);
        cudaStreamSynchronize(stream);

        auto result = d_output->get_host_vector<float>();
        std::cout << "Work queue pattern completed!" << std::endl;
        std::cout << "First result value: " << result[0] << std::endl;
    }

    cudaStreamDestroy(stream);
}

int main()
{
    std::cout << "Advanced CUDA Graph Examples" << std::endl;
    std::cout << "============================" << std::endl;

    if (cuda_init() != 0) {
        std::cerr << "Failed to initialize CUDA!" << std::endl;
        return -1;
    }

    try {
        advanced_cuda_graph_examples();
        std::cout << "\nAll advanced examples completed successfully!"
                  << std::endl;
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
