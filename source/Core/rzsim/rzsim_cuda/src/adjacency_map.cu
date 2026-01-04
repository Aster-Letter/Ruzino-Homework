#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <Eigen/Eigen>
#include <RHI/cuda.hpp>
#include <RHI/rhi.hpp>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>

#include "RHI/internal/cuda_extension.hpp"
#include "rzsim_cuda/adjacency_map.cuh"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

// Directed edge representation used for sorting/unique
struct Edge {
    unsigned src;
    unsigned dst;
};

// Build directed edges for each face (two per undirected edge)
__global__ void build_edges_kernel(
    const int* face_vertex_counts,
    const int* face_vertex_indices,
    const unsigned* face_offsets,
    Edge* edges,
    unsigned num_faces)
{
    unsigned face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces)
        return;

    int vertex_count = face_vertex_counts[face_idx];
    unsigned face_start = face_offsets[face_idx];
    unsigned edge_base = 2 * face_start;

    for (int i = 0; i < vertex_count; i++) {
        unsigned v0 = face_vertex_indices[face_start + i];
        unsigned v1 = face_vertex_indices[face_start + (i + 1) % vertex_count];

        edges[edge_base + 2 * i] = { v0, v1 };      // v0 -> v1
        edges[edge_base + 2 * i + 1] = { v1, v0 };  // v1 -> v0
    }
}

// Count degree (unique neighbors) per vertex
__global__ void count_degrees_kernel(
    const Edge* edges,
    unsigned edge_count,
    unsigned* neighbor_counts)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edge_count)
        return;

    unsigned src = edges[idx].src;
    atomicAdd(&neighbor_counts[src], 1);
}

// Fill adjacency list from unique edges
__global__ void fill_adjacency_from_edges_kernel(
    const Edge* edges,
    unsigned edge_count,
    unsigned* neighbor_write_pos,
    const unsigned* offsets,
    unsigned* adjacency_list)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edge_count)
        return;

    unsigned src = edges[idx].src;
    unsigned dst = edges[idx].dst;

    unsigned offset = offsets[src];
    unsigned pos = atomicAdd(&neighbor_write_pos[src], 1);
    adjacency_list[offset + 1 + pos] = dst;
}

std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
compute_adjacency_map_gpu(
    cuda::CUDALinearBufferHandle vertices,
    cuda::CUDALinearBufferHandle faceVertexCounts,
    cuda::CUDALinearBufferHandle faceVertexIndices)
{
    auto vertex_buffer_addr = vertices->get_device_ptr();
    auto face_vertex_counts_addr = faceVertexCounts->get_device_ptr();
    auto face_vertex_indices_addr = faceVertexIndices->get_device_ptr();

    auto vertex_count = vertices->getDesc().element_count;
    auto face_count = faceVertexCounts->getDesc().element_count;
    thrust::device_ptr<const int> counts_ptr(
        reinterpret_cast<const int*>(face_vertex_counts_addr));
    thrust::device_ptr<const int> indices_ptr(
        reinterpret_cast<const int*>(face_vertex_indices_addr));

    // Prefix for face vertex indices to avoid per-face loops
    thrust::device_vector<unsigned> face_offsets(face_count);
    thrust::exclusive_scan(
        thrust::device,
        counts_ptr,
        counts_ptr + face_count,
        face_offsets.begin());

    // Total directed edges = 2 * sum(faceVertexCounts)
    unsigned total_face_vertices =
        thrust::reduce(thrust::device, counts_ptr, counts_ptr + face_count, 0);
    unsigned total_directed_edges = 2 * total_face_vertices;

    thrust::device_vector<Edge> edges(total_directed_edges);

    int threads_per_block = 256;
    int face_blocks = (face_count + threads_per_block - 1) / threads_per_block;

    build_edges_kernel<<<face_blocks, threads_per_block>>>(
        (const int*)face_vertex_counts_addr,
        (const int*)face_vertex_indices_addr,
        thrust::raw_pointer_cast(face_offsets.data()),
        thrust::raw_pointer_cast(edges.data()),
        face_count);
    cudaDeviceSynchronize();

    // Radix-sort edges by 64-bit key (src<<32 | dst) then unique
    thrust::device_vector<unsigned long long> keys(total_directed_edges);
    thrust::transform(
        thrust::device,
        edges.begin(),
        edges.end(),
        keys.begin(),
        [] __device__(const Edge& e) {
            return (static_cast<unsigned long long>(e.src) << 32) |
                   static_cast<unsigned long long>(e.dst);
        });

    thrust::sort_by_key(
        thrust::device, keys.begin(), keys.end(), edges.begin());

    auto unique_pair = thrust::unique_by_key(
        thrust::device, keys.begin(), keys.end(), edges.begin());
    keys.erase(unique_pair.first, keys.end());
    edges.erase(unique_pair.second, edges.end());

    unsigned unique_edge_count = static_cast<unsigned>(edges.size());

    // Degree per vertex after dedup
    thrust::device_vector<unsigned> neighbor_counts(vertex_count, 0);
    int edge_blocks =
        (unique_edge_count + threads_per_block - 1) / threads_per_block;

    count_degrees_kernel<<<edge_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(edges.data()),
        unique_edge_count,
        thrust::raw_pointer_cast(neighbor_counts.data()));
    cudaDeviceSynchronize();

    // sizes = degree + 1 (store count first)
    thrust::device_vector<unsigned> sizes(vertex_count);
    thrust::transform(
        neighbor_counts.begin(),
        neighbor_counts.end(),
        sizes.begin(),
        [] __device__(unsigned count) { return count + 1; });

    // Offsets via CUB exclusive scan
    thrust::device_vector<unsigned> offsets(vertex_count);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        thrust::raw_pointer_cast(sizes.data()),
        thrust::raw_pointer_cast(offsets.data()),
        vertex_count);

    thrust::device_vector<char> temp_storage(temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(
        thrust::raw_pointer_cast(temp_storage.data()),
        temp_storage_bytes,
        thrust::raw_pointer_cast(sizes.data()),
        thrust::raw_pointer_cast(offsets.data()),
        vertex_count);
    cudaDeviceSynchronize();

    unsigned total_size = thrust::reduce(sizes.begin(), sizes.end(), 0u);

    cuda::CUDALinearBufferDesc desc;
    desc.element_count = total_size;
    desc.element_size = sizeof(unsigned);

    auto result_buffer = cuda::create_cuda_linear_buffer(desc);
    auto result_ptr = (unsigned*)result_buffer->get_device_ptr();

    // Write counts at offsets
    thrust::for_each(
        thrust::device,
        thrust::make_zip_iterator(
            thrust::make_tuple(offsets.begin(), neighbor_counts.begin())),
        thrust::make_zip_iterator(
            thrust::make_tuple(offsets.end(), neighbor_counts.end())),
        [result_ptr] __device__(const auto& tuple) {
            unsigned offset = thrust::get<0>(tuple);
            unsigned count = thrust::get<1>(tuple);
            result_ptr[offset] = count;
        });

    // Fill neighbors
    thrust::device_vector<unsigned> neighbor_write_pos(vertex_count, 0);
    fill_adjacency_from_edges_kernel<<<edge_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(edges.data()),
        unique_edge_count,
        thrust::raw_pointer_cast(neighbor_write_pos.data()),
        thrust::raw_pointer_cast(offsets.data()),
        result_ptr);
    cudaDeviceSynchronize();

    // Create offset buffer for random access
    cuda::CUDALinearBufferDesc offset_desc;
    offset_desc.element_count = vertex_count;
    offset_desc.element_size = sizeof(unsigned);

    auto offset_buffer = cuda::create_cuda_linear_buffer(offset_desc);
    auto offset_ptr = (unsigned*)offset_buffer->get_device_ptr();

    cudaMemcpy(
        offset_ptr,
        thrust::raw_pointer_cast(offsets.data()),
        vertex_count * sizeof(unsigned),
        cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    return std::make_tuple(result_buffer, offset_buffer);
}

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
