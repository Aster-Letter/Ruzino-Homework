#pragma once

#include <RHI/cuda.hpp>
#include <memory>
#include <vector>

#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

// Forward declarations
class Geometry;

// Add your API functions here

// Legacy interface - returns only adjacency list
RZSIM_API std::tuple<cuda::CUDALinearBufferHandle, cuda::CUDALinearBufferHandle>
get_adjcency_map_gpu(const Geometry& g);

RZSIM_API std::tuple<std::vector<unsigned>, std::vector<unsigned>>
get_adjcency_map(const Geometry& g);

// New interface - returns both adjacency list and offset buffer
// adjacency_buffer: [count_v0, neighbor1, neighbor2, ... | count_v1, neighbor1,
// neighbor2, ... | ...] offset_buffer: offset_buffer[vertex_id] points to
// vertex_id's data in adjacency_buffer
// struct AdjacencyMapResult {
//     cuda::CUDALinearBufferHandle adjacency_list;
//     cuda::CUDALinearBufferHandle offset_buffer;
// };

// RZSIM_API AdjacencyMapResult
// get_adjcency_map_with_offsets_gpu(const Geometry& g);

RUZINO_NAMESPACE_CLOSE_SCOPE
