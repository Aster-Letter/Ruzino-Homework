#include <sstream>
#if RUZINO_WITH_CUDA

#include <RHI/internal/cuda_extension_utils.h>

#include <RHI/internal/cuda_graph.hpp>
#include <iostream>

RUZINO_NAMESPACE_OPEN_SCOPE

namespace cuda {

class CUDAGraph : public nvrhi::RefCounter<ICUDAGraph> {
   public:
    CUDAGraph(const CUDAGraphDesc& desc) : desc_(desc), capturing_(false)
    {
        CUDA_CHECK(cudaGraphCreate(&graph_, 0));
    }

    ~CUDAGraph()
    {
        if (graph_) {
            cudaGraphDestroy(graph_);
        }
    }

    const CUDAGraphDesc& getDesc() const override
    {
        return desc_;
    }
    bool beginCapture() override
    {
        if (capturing_) {
            std::cerr << "Graph is already capturing!" << std::endl;
            return false;
        }

        cudaStream_t stream = desc_.stream;
        if (!stream) {
            CUDA_CHECK(cudaStreamCreate(&stream));
            // Need to create a mutable copy to modify
            const_cast<CUDAGraphDesc&>(desc_).stream = stream;
        }

        // 检查流是否已经在捕获模式
        cudaStreamCaptureStatus captureStatus;
        cudaError_t result =
            cudaStreamGetCaptureInfo(stream, &captureStatus, nullptr);
        if (result == cudaSuccess &&
            captureStatus != cudaStreamCaptureStatusNone) {
            std::cerr << "Stream is already capturing! Status: "
                      << captureStatus << std::endl;
            return false;
        }

        result = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (result != cudaSuccess) {
            std::cerr << "Failed to begin capture: "
                      << cudaGetErrorString(result) << std::endl;
            return false;
        }

        capturing_ = true;
        return true;
    }

    CUDAGraphExecHandle endCapture() override;

    bool isCapturing() const override
    {
        return capturing_;
    }

    cudaGraph_t getCudaGraph() const override
    {
        return graph_;
    }

   private:
    CUDAGraphDesc desc_;
    cudaGraph_t graph_;
    bool capturing_;
};

class CUDAGraphExec : public nvrhi::RefCounter<ICUDAGraphExec> {
   public:
    CUDAGraphExec(cudaGraphExec_t exec, cudaStream_t stream)
        : exec_(exec),
          stream_(stream)
    {
    }

    ~CUDAGraphExec()
    {
        if (exec_) {
            cudaGraphExecDestroy(exec_);
        }
    }

    bool launch(cudaStream_t stream = nullptr) override
    {
        cudaStream_t launchStream = stream ? stream : stream_;

        cudaError_t result = cudaGraphLaunch(exec_, launchStream);
        if (result != cudaSuccess) {
            std::cerr << "Failed to launch graph: "
                      << cudaGetErrorString(result) << std::endl;
            return false;
        }

        return true;
    }
    bool update(CUDAGraphHandle newGraph) override
    {
        cudaGraphExecUpdateResultInfo resultInfo;

        cudaError_t result =
            cudaGraphExecUpdate(exec_, newGraph->getCudaGraph(), &resultInfo);

        if (result != cudaSuccess ||
            resultInfo.result != cudaGraphExecUpdateSuccess) {
            std::cerr << "Failed to update graph" << std::endl;
            return false;
        }

        return true;
    }

    cudaGraphExec_t getCudaGraphExec() const override
    {
        return exec_;
    }

   private:
    cudaGraphExec_t exec_;
    cudaStream_t stream_;
};

CUDAGraphExecHandle CUDAGraph::endCapture()
{
    if (!capturing_) {
        std::cerr << "Graph is not capturing!" << std::endl;
        return nullptr;
    }

    cudaGraph_t capturedGraph;
    cudaError_t result = cudaStreamEndCapture(desc_.stream, &capturedGraph);
    if (result != cudaSuccess) {
        std::cerr << "Failed to end capture: " << cudaGetErrorString(result)
                  << std::endl;
        return nullptr;
    }

    capturing_ = false;

    // Create executable graph
    cudaGraphExec_t graphExec;
    result =
        cudaGraphInstantiate(&graphExec, capturedGraph, nullptr, nullptr, 0);
    if (result != cudaSuccess) {
        std::cerr << "Failed to instantiate graph: "
                  << cudaGetErrorString(result) << std::endl;
        cudaGraphDestroy(capturedGraph);
        return nullptr;
    }

    // Copy the captured graph to our internal graph
    cudaGraphDestroy(graph_);
    graph_ = capturedGraph;

    return nvrhi::RefCountPtr<ICUDAGraphExec>(
        new CUDAGraphExec(graphExec, desc_.stream));
}

// API implementations
CUDAGraphHandle create_cuda_graph(const CUDAGraphDesc& desc)
{
    return nvrhi::RefCountPtr<ICUDAGraph>(new CUDAGraph(desc));
}

CUDAGraphHandle create_cuda_graph(cudaStream_t stream)
{
    CUDAGraphDesc desc(stream);
    return create_cuda_graph(desc);
}

CUDAGraphCapture::~CUDAGraphCapture()
{
    finalize();
}

CUDAGraphCapture& CUDAGraphCapture::operator=(CUDAGraphCapture&& other) noexcept
{
    if (this != &other) {
        finalize();
        graph_ = std::move(other.graph_);
        exec_ = std::move(other.exec_);
        capturing_ = other.capturing_;
        other.capturing_ = false;
    }
    return *this;
}

CUDAGraphExecHandle CUDAGraphCapture::get()
{
    finalize();
    return exec_;
}

ICUDAGraphExec* CUDAGraphCapture::operator->()
{
    finalize();
    return exec_.Get();
}

ICUDAGraphExec& CUDAGraphCapture::operator*()
{
    finalize();
    return *exec_;
}

CUDAGraphCapture::operator nvrhi::RefCountPtr<ICUDAGraphExec>()
{
    return get();
}

CUDAGraphExecHandle CUDAGraphCapture::release()
{
    finalize();
    return std::move(exec_);
}

void CUDAGraphCapture::finalize()
{
    if (capturing_ && graph_ && graph_->isCapturing()) {
        exec_ = graph_->endCapture();
        capturing_ = false;
    }
}

CUDAGraphBuilder& CUDAGraphBuilder::addMemAllocNode(
    void** ptr,
    size_t size,
    const std::vector<cudaGraphNode_t>& deps)
{
    cudaMemAllocNodeParams params = {};
    params.bytesize = size;
    params.poolProps.allocType = cudaMemAllocationTypePinned;
    params.poolProps.location.type = cudaMemLocationTypeDevice;
    params.poolProps.location.id = 0;

    // Use memory pool if available
    if (memPool_) {
        params.poolProps.handleTypes = cudaMemHandleTypeNone;
        // Associate with the memory pool
        // Note: For simplicity, we'll use the default device allocation
        // In a real implementation, you'd properly configure pool allocation
    }

    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddMemAllocNode(
        &node, graph_, deps.data(), deps.size(), &params));
    *ptr = params.dptr;
    nodes_.push_back(node);
    return *this;
}

CUDAGraphBuilder& CUDAGraphBuilder::addMemFreeNode(
    void* ptr,
    const std::vector<cudaGraphNode_t>& deps)
{
    cudaGraphNode_t node;
    CUDA_CHECK(
        cudaGraphAddMemFreeNode(&node, graph_, deps.data(), deps.size(), ptr));
    nodes_.push_back(node);
    return *this;
}

CUDAGraphBuilder& CUDAGraphBuilder::addMemsetNode(
    void* ptr,
    int value,
    size_t count,
    const std::vector<cudaGraphNode_t>& deps)
{
    cudaMemsetParams params = {};
    params.dst = ptr;
    params.value = value;
    params.elementSize = 1;
    params.width = count;
    params.height = 1;

    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddMemsetNode(
        &node, graph_, deps.data(), deps.size(), &params));
    nodes_.push_back(node);
    return *this;
}

CUDAGraphBuilder& CUDAGraphBuilder::addHostNode(
    cudaHostFn_t fn,
    void* userData,
    const std::vector<cudaGraphNode_t>& deps)
{
    cudaHostNodeParams params = {};
    params.fn = fn;
    params.userData = userData;

    cudaGraphNode_t node;
    CUDA_CHECK(
        cudaGraphAddHostNode(&node, graph_, deps.data(), deps.size(), &params));
    nodes_.push_back(node);
    return *this;
}

CUDAGraphExecHandle CUDAGraphBuilder::build()
{
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph_, nullptr, nullptr, 0));

    // 这里需要创建一个wrapper，简化处理
    auto handle = nvrhi::RefCountPtr<ICUDAGraphExec>(
        new CUDAGraphExec(graphExec, nullptr));  // 需要实现

    return handle;
}

CUDAGraphMemoryPool::CUDAGraphMemoryPool(size_t initialSize)
{
    cudaMemPoolProps poolProps = {};
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.handleTypes = cudaMemHandleTypeNone;
    poolProps.location.type = cudaMemLocationTypeDevice;
    poolProps.location.id = 0;

    CUDA_CHECK(cudaMemPoolCreate(&pool_, &poolProps));

    // Set initial threshold
    uint64_t threshold = initialSize;
    CUDA_CHECK(cudaMemPoolSetAttribute(
        pool_, cudaMemPoolAttrReleaseThreshold, &threshold));
}
}  // namespace cuda

RUZINO_NAMESPACE_CLOSE_SCOPE

#endif  // RUZINO_WITH_CUDA
