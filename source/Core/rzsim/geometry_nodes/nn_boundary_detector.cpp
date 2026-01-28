#include <memory>

#include "GCore/Components/MeshComponent.h"
#include "GCore/geom_payload.hpp"
#include "RHI/cuda.hpp"
#include "nodes/core/def/node_def.hpp"
#include "rzpython/rzpython.hpp"

using namespace Ruzino;

NODE_DEF_OPEN_SCOPE

// Storage for caching GPU buffers and detection results
struct NNBoundaryDetectorStorage {
    cuda::CUDALinearBufferHandle vertices_buffer;
    bool initialized = false;
    int num_vertices = 0;

    // Cache for detection results
    std::vector<glm::vec3> cached_vertices;
    float cached_shape_code = -1.0f;
    bool cached_has_shape_code_2 = false;
    float cached_shape_code_2 = -1.0f;
    float cached_distance_threshold = -1.0f;
    std::string cached_model_path;
    std::vector<float> cached_boundary_values;
    bool cache_valid = false;

    constexpr static bool has_storage = false;
};

NODE_DECLARATION_FUNCTION(nn_boundary_detector)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<float>("Shape Code").default_val(0.0f).min(0.0f).max(1.0f);
    b.add_input<bool>("Has Shape Code 2").default_val(false);
    b.add_input<float>("Shape Code 2").default_val(0.0f).min(0.0f).max(1.0f);
    b.add_input<float>("Distance Threshold")
        .default_val(1e-6f)
        .min(1e-8f)
        .max(1e-3f);
    b.add_input<std::string>("Model Path")
        .default_val("mesh_growing_y_neg_20260119_015826");
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(nn_boundary_detector)
{
    spdlog::info("[NNBoundaryDetector] Node execution started");

    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();
    auto& storage = params.get_storage<NNBoundaryDetectorStorage&>();

    float shape_code = params.get_input<float>("Shape Code");
    bool has_shape_code_2 = params.get_input<bool>("Has Shape Code 2");
    float shape_code_2 = params.get_input<float>("Shape Code 2");
    float distance_threshold = params.get_input<float>("Distance Threshold");
    std::string model_path = params.get_input<std::string>("Model Path");

    // Get vertices from geometry
    auto mesh_component = input_geom.get_component<MeshComponent>();
    if (!mesh_component) {
        spdlog::warn("No mesh component found in geometry");
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    std::vector<glm::vec3> vertices = mesh_component->get_vertices();

    // Check if we can use cached results
    bool inputs_changed =
        !storage.cache_valid || storage.cached_shape_code != shape_code ||
        storage.cached_has_shape_code_2 != has_shape_code_2 ||
        storage.cached_shape_code_2 != shape_code_2 ||
        storage.cached_distance_threshold != distance_threshold ||
        storage.cached_vertices.size() != vertices.size() ||
        storage.cached_vertices != vertices ||
        storage.cached_model_path != model_path;

    if (!inputs_changed) {
        spdlog::info(
            "[NNBoundaryDetector] Using cached boundary detection results");

        // Use cached results
        mesh_component->add_vertex_scalar_quantity(
            "nn_dirichlet", storage.cached_boundary_values);

        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    // Inputs changed, need to recompute
    spdlog::info("[NNBoundaryDetector] Inputs changed, running detection...");

    try {
        // Import torch and deducer module
        python::call<void>("import torch");
        python::call<void>(
            "import sys\n"
            "sys.path.insert(0, "
            "r'C:"
            "\\Users\\Pengfei\\WorkSpace\\Ruzino\\source\\Core\\rzsim\\geometry"
            "_nodes')\n"
            "import deducer");

        // Initialize basis set
        python::send("model_name", model_path);
        std::string init_result = python::call<std::string>(
            "deducer.initialize_basis_set(model_name)");
        python::flush_python_output();  // Flush Python print output
        spdlog::info("Basis set initialization: {}", init_result);

        // Create or update GPU buffer for vertices
        if (!storage.initialized || storage.num_vertices != vertices.size()) {
            storage.vertices_buffer = cuda::create_cuda_linear_buffer(vertices);
            storage.num_vertices = vertices.size();
            storage.initialized = true;
        }
        else {
            // Update existing buffer
            storage.vertices_buffer->assign_host_vector(vertices);
        }

        spdlog::info(
            "Detecting Dirichlet boundary on {} vertices with "
            "shape_code={:.2f}, threshold={:.2e}",
            vertices.size(),
            shape_code,
            distance_threshold);

        // Zero-copy send CUDA array to Python
        float* gpu_ptr = storage.vertices_buffer->get_device_ptr<float>();
        python::send(
            "vertices_cuda",
            python::CudaArrayView<float>{ gpu_ptr, { vertices.size(), 3 } });

        // Send shape code and threshold to Python
        if (has_shape_code_2) {
            // Send two shape codes as a list
            std::vector<float> shape_codes = { shape_code, shape_code_2 };
            python::send("shape_code_values", shape_codes);
            spdlog::info(
                "Using dual shape codes: [{:.2f}, {:.2f}]",
                shape_code,
                shape_code_2);
        }
        else {
            // Send single shape code
            python::send("shape_code_value", shape_code);
        }

        python::send("distance_threshold", distance_threshold);

        // Call Python detection function
        std::string detection_code =
            "import numpy as np\n"
            "import torch\n";

        if (has_shape_code_2) {
            detection_code +=
                "boundary_result = "
                "deducer.detect_dirichlet_boundary("
                "vertices_cuda, "
                "shape_code_value=shape_code_values, "
                "distance_threshold=distance_threshold)";
        }
        else {
            detection_code +=
                "boundary_result = "
                "deducer.detect_dirichlet_boundary("
                "vertices_cuda, "
                "shape_code_value=shape_code_value, "
                "distance_threshold=distance_threshold)";
        }

        python::call<void>(detection_code);
        python::flush_python_output();  // Flush Python print output

        // Retrieve results as vector<float>
        std::vector<float> boundary_values =
            python::call<std::vector<float>>("boundary_result.tolist()");

        if (boundary_values.size() != vertices.size()) {
            spdlog::error(
                "Boundary detection result size ({}) does not match vertex "
                "count ({})",
                boundary_values.size(),
                vertices.size());
            params.set_output<Geometry>("Geometry", std::move(input_geom));
            return true;
        }

        // Count boundary vertices
        int boundary_count = std::count_if(
            boundary_values.begin(), boundary_values.end(), [](float v) {
                return v > 0.5f;
            });

        spdlog::info(
            "Detected {} / {} vertices on Dirichlet boundary",
            boundary_count,
            vertices.size());

        // Update cache
        storage.cached_vertices = vertices;
        storage.cached_shape_code = shape_code;
        storage.cached_has_shape_code_2 = has_shape_code_2;
        storage.cached_shape_code_2 = shape_code_2;
        storage.cached_distance_threshold = distance_threshold;
        storage.cached_model_path = model_path;
        storage.cached_boundary_values = boundary_values;
        storage.cache_valid = true;
        spdlog::info(
            "[NNBoundaryDetector] Cache updated (model={})", model_path);

        // Add boundary detection as vertex scalar quantity
        mesh_component->add_vertex_scalar_quantity(
            "nn_dirichlet", boundary_values);
        spdlog::info("Added 'nn_dirichlet' vertex attribute");
    }
    catch (const std::exception& e) {
        spdlog::error("Python call failed: {}", e.what());
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(nn_boundary_detector);

NODE_DEF_CLOSE_SCOPE
