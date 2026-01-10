#include "rzsim/rzsim.h"

#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "rzsim_cuda/adjacency_map.cuh"

RUZINO_NAMESPACE_OPEN_SCOPE

// Surface mesh adjacency (for triangles)
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
get_surface_adjacency_gpu(const Geometry& g)
{
    auto mesh = g.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto faceVertexIndices = mesh->get_face_vertex_indices();
    
    auto vertex_buffer = cuda::create_cuda_linear_buffer(vertices);
    auto triangle_buffer = cuda::create_cuda_linear_buffer(faceVertexIndices);
    
    return rzsim_cuda::compute_surface_adjacency_gpu(vertex_buffer, triangle_buffer);
}

std::tuple<std::vector<unsigned>, std::vector<unsigned>>
get_surface_adjacency(const Geometry& g)
{
    auto [adjacency_buffer, offset_buffer] = get_surface_adjacency_gpu(g);
    
    auto adjacency_cpu = adjacency_buffer->get_host_vector<unsigned>();
    auto offset_cpu = offset_buffer->get_host_vector<unsigned>();
    
    return std::make_tuple(adjacency_cpu, offset_cpu);
}

// Volume mesh adjacency (for tetrahedra)
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
get_volume_adjacency_gpu(const Geometry& g)
{
    auto mesh = g.get_component<MeshComponent>();
    auto vertices = mesh->get_vertices();
    auto faceVertexIndices = mesh->get_face_vertex_indices();
    
    auto vertex_buffer = cuda::create_cuda_linear_buffer(vertices);
    auto triangle_buffer = cuda::create_cuda_linear_buffer(faceVertexIndices);
    
    auto [adjacency, offsets, num_elements] = rzsim_cuda::compute_volume_adjacency_gpu(vertex_buffer, triangle_buffer);
    // Discard num_elements for backward compatibility
    return std::make_tuple(adjacency, offsets);
}

std::tuple<std::vector<unsigned>, std::vector<unsigned>>
get_volume_adjacency(const Geometry& g)
{
    auto [adjacency_buffer, offset_buffer] = get_volume_adjacency_gpu(g);
    
    auto adjacency_cpu = adjacency_buffer->get_host_vector<unsigned>();
    auto offset_cpu = offset_buffer->get_host_vector<unsigned>();
    
    return std::make_tuple(adjacency_cpu, offset_cpu);
}

RUZINO_NAMESPACE_CLOSE_SCOPE
