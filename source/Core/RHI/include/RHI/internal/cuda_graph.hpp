#pragma once

#include "RHI/internal/cuda_extension_utils.h"
#if RUZINO_WITH_CUDA

#include <RHI/api.h>
#include <cuda_runtime.h>
#include <nvrhi/nvrhi.h>

RUZINO_NAMESPACE_OPEN_SCOPE

namespace cuda {

// Forward declarations
class ICUDAGraph;
class ICUDAGraphExec;
class CUDAGraphCapture;

using CUDAGraphHandle = nvrhi::RefCountPtr<ICUDAGraph>;
using CUDAGraphExecHandle = nvrhi::RefCountPtr<ICUDAGraphExec>;

// CUDA Graph description
struct CUDAGraphDesc {
    cudaStream_t stream = nullptr;  // Stream to capture operations on
    bool enable_capture = true;     // Whether to enable graph capture

    CUDAGraphDesc() = default;
    CUDAGraphDesc(cudaStream_t s) : stream(s)
    {
    }
};

// CUDA Graph interface
class ICUDAGraph : public nvrhi::IResource {
   public:
    virtual ~ICUDAGraph() = default;

    [[nodiscard]] virtual const CUDAGraphDesc& getDesc() const = 0;

    // Begin capturing operations into the graph
    virtual bool beginCapture() = 0;

    // End capturing and create executable graph
    virtual CUDAGraphExecHandle endCapture() = 0;

    // Check if currently capturing
    virtual bool isCapturing() const = 0;

    // Get the underlying CUDA graph
    virtual cudaGraph_t getCudaGraph() const = 0;
};

// CUDA Graph Executable interface
class ICUDAGraphExec : public nvrhi::IResource {
   public:
    virtual ~ICUDAGraphExec() = default;

    // Launch the graph
    virtual bool launch(cudaStream_t stream = nullptr) = 0;

    // Update the graph with a new graph (if topology is compatible)
    virtual bool update(CUDAGraphHandle newGraph) = 0;

    // Get the underlying CUDA graph executable
    virtual cudaGraphExec_t getCudaGraphExec() const = 0;
};

// API Functions
RHI_API CUDAGraphHandle create_cuda_graph(const CUDAGraphDesc& desc);

RHI_API CUDAGraphHandle create_cuda_graph(cudaStream_t stream = nullptr);

// Elegant RAII wrapper that IS the captured graph executable
class RHI_API CUDAGraphCapture {
   public:
    // Constructor starts capture
    CUDAGraphCapture(CUDAGraphHandle graph) : graph_(graph), exec_(nullptr)
    {
        if (graph_ && graph_->beginCapture()) {
            capturing_ = true;
        }
        else {
            capturing_ = false;
        }
    }

    // Destructor ensures capture is ended
    ~CUDAGraphCapture();

    // Move constructor
    CUDAGraphCapture(CUDAGraphCapture&& other) noexcept
        : graph_(std::move(other.graph_)),
          exec_(std::move(other.exec_)),
          capturing_(other.capturing_)
    {
        other.capturing_ = false;
    }

    // Move assignment
    CUDAGraphCapture& operator=(CUDAGraphCapture&& other) noexcept;

    // Delete copy operations
    CUDAGraphCapture(const CUDAGraphCapture&) = delete;
    CUDAGraphCapture& operator=(const CUDAGraphCapture&) = delete;

    // Check if capture was successful
    explicit operator bool() const
    {
        return capturing_ || exec_;
    }

    // Get the executable (automatically finalizes capture)
    CUDAGraphExecHandle get();

    // Dereference operators for direct access
    ICUDAGraphExec* operator->();

    ICUDAGraphExec& operator*();

    // Implicit conversion to handle
    operator CUDAGraphExecHandle();

    // Release ownership of the executable
    CUDAGraphExecHandle release();

   private:
    void finalize();

   private:
    CUDAGraphHandle graph_;
    CUDAGraphExecHandle exec_;
    bool capturing_ = false;
};

// Ultra-elegant RAII factory function that returns a ready-to-use executable
template<typename F>
CUDAGraphCapture capture_graph(cudaStream_t stream, F&& operations)
{
    auto graph = create_cuda_graph(stream);
    CUDAGraphCapture capture(graph);

    if (capture) {
        operations();
    }

    return capture;  // Move semantics, finalization happens on first access
}

// Even more elegant - direct lambda capture
template<typename F>
auto with_cuda_graph(cudaStream_t stream, F&& operations) -> CUDAGraphCapture
{
    return capture_graph(stream, std::forward<F>(operations));
}

// Advanced graph builder for explicit node management
class RHI_API CUDAGraphBuilder {
   public:
    CUDAGraphBuilder(cudaMemPool_t pool = nullptr) : memPool_(pool)
    {
        CUDA_CHECK(cudaGraphCreate(&graph_, 0));
    }

    ~CUDAGraphBuilder()
    {
        if (graph_) {
            cudaGraphDestroy(graph_);
        }
    }

    // Add memory allocation node
    CUDAGraphBuilder& addMemAllocNode(
        void** ptr,
        size_t size,
        const std::vector<cudaGraphNode_t>& deps = {});

    // Add memory free node
    CUDAGraphBuilder& addMemFreeNode(
        void* ptr,
        const std::vector<cudaGraphNode_t>& deps = {});

    // Add memset node
    CUDAGraphBuilder& addMemsetNode(
        void* ptr,
        int value,
        size_t count,
        const std::vector<cudaGraphNode_t>& deps = {});

    // Add kernel node
    template<typename Kernel, typename... Args>
    CUDAGraphBuilder& addKernelNode(
        Kernel kernel,
        dim3 grid,
        dim3 block,
        size_t sharedMem,
        const std::vector<cudaGraphNode_t>& deps,
        Args... args)
    {
        cudaKernelNodeParams params = {};

        // 这里需要处理参数打包，简化版本
        static void* kernelArgs[] = { &args... };

        params.func = (void*)kernel;
        params.gridDim = grid;
        params.blockDim = block;
        params.sharedMemBytes = sharedMem;
        params.kernelParams = kernelArgs;

        cudaGraphNode_t node;
        CUDA_CHECK(cudaGraphAddKernelNode(
            &node, graph_, deps.data(), deps.size(), &params));
        nodes_.push_back(node);
        return *this;
    }

    // Add host function node
    CUDAGraphBuilder& addHostNode(
        cudaHostFn_t fn,
        void* userData,
        const std::vector<cudaGraphNode_t>& deps = {});

    // Build the final executable graph
    CUDAGraphExecHandle build();

    // Get specific node for dependencies
    cudaGraphNode_t getNode(size_t index) const
    {
        return (index < nodes_.size()) ? nodes_[index] : nullptr;
    }

    cudaGraphNode_t getLastNode() const
    {
        return nodes_.empty() ? nullptr : nodes_.back();
    }

   private:
    cudaGraph_t graph_;
    std::vector<cudaGraphNode_t> nodes_;
    cudaMemPool_t memPool_;
};

// Factory function for advanced graph building
inline CUDAGraphBuilder create_advanced_graph()
{
    return CUDAGraphBuilder();
}

// Factory function for advanced graph building with memory pool
inline CUDAGraphBuilder create_advanced_graph_with_pool(cudaMemPool_t pool)
{
    return CUDAGraphBuilder(pool);
}

// Memory pool management for graphs
class RHI_API CUDAGraphMemoryPool {
   public:
    CUDAGraphMemoryPool(size_t initialSize = 1024 * 1024);

    ~CUDAGraphMemoryPool()
    {
        if (pool_) {
            cudaMemPoolDestroy(pool_);
        }
    }

    cudaMemPool_t getPool() const
    {
        return pool_;
    }

   private:
    cudaMemPool_t pool_;
};

}  // namespace cuda

RUZINO_NAMESPACE_CLOSE_SCOPE

#endif  // RUZINO_WITH_CUDA
