#pragma once

#include <RHI/cuda.hpp>

#include "RHI/internal/cuda_extension.hpp"
#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace rzsim_cuda {

struct CSRMatrix {
    cuda::CUDALinearBufferHandle row_offsets;
    cuda::CUDALinearBufferHandle col_indices;
    cuda::CUDALinearBufferHandle values;
    int num_rows;
    int num_cols;
    int nnz;
};

RZSIM_CUDA_API
cuda::CUDALinearBufferHandle build_edge_set_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle edges);

// Build per-vertex spring adjacency for efficient gradient/hessian computation
// Returns: (spring_indices_per_vertex, vertex_spring_offsets)
// Format: spring_indices_per_vertex[vertex_spring_offsets[v]..vertex_spring_offsets[v+1]] = spring indices for vertex v
RZSIM_CUDA_API
std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
build_vertex_spring_adjacency_gpu(
    cuda::CUDALinearBufferHandle springs,
    int num_particles);

RZSIM_CUDA_API
void explicit_step_gpu(
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle v,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle x_tilde);

RZSIM_CUDA_API
void setup_external_forces_gpu(
    float mass,
    float gravity,
    int num_particles,
    cuda::CUDALinearBufferHandle f_ext);

RZSIM_CUDA_API
cuda::CUDALinearBufferHandle compute_rest_lengths_gpu(
    cuda::CUDALinearBufferHandle positions,
    cuda::CUDALinearBufferHandle springs);

RZSIM_CUDA_API
void compute_gradient_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle springs,
    cuda::CUDALinearBufferHandle rest_lengths,
    cuda::CUDALinearBufferHandle spring_indices_per_vertex,
    cuda::CUDALinearBufferHandle vertex_spring_offsets,
    float stiffness,
    float dt,
    int num_particles,
    cuda::CUDALinearBufferHandle grad);

RZSIM_CUDA_API
CSRMatrix assemble_hessian_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle springs,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles);

RZSIM_CUDA_API
float compute_energy_gpu(
    cuda::CUDALinearBufferHandle x_curr,
    cuda::CUDALinearBufferHandle x_tilde,
    cuda::CUDALinearBufferHandle M_diag,
    cuda::CUDALinearBufferHandle f_ext,
    cuda::CUDALinearBufferHandle springs,
    cuda::CUDALinearBufferHandle rest_lengths,
    float stiffness,
    float dt,
    int num_particles);

// GPU vector operations to avoid CPU-GPU transfers
RZSIM_CUDA_API
float compute_vector_norm_gpu(
    cuda::CUDALinearBufferHandle vec,
    int size);

RZSIM_CUDA_API
float compute_dot_product_gpu(
    cuda::CUDALinearBufferHandle vec1,
    cuda::CUDALinearBufferHandle vec2,
    int size);

RZSIM_CUDA_API
void axpy_gpu(
    float alpha,
    cuda::CUDALinearBufferHandle x,
    cuda::CUDALinearBufferHandle y,
    cuda::CUDALinearBufferHandle result,
    int size);

RZSIM_CUDA_API
void negate_gpu(
    cuda::CUDALinearBufferHandle in,
    cuda::CUDALinearBufferHandle out,
    int size);

RZSIM_CUDA_API
void project_to_ground_gpu(
    cuda::CUDALinearBufferHandle positions,
    int num_particles,
    float ground_height);

}  // namespace rzsim_cuda

RUZINO_NAMESPACE_CLOSE_SCOPE
