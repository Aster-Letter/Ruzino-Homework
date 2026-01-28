#include <memory>

#include "GCore/Components/MeshComponent.h"
#include "GCore/geom_payload.hpp"
#include "RHI/cuda.hpp"
#include "nodes/core/def/node_def.hpp"
#include "rzpython/rzpython.hpp"
#include "rzsim/reduced_order_basis.h"

using namespace Ruzino;

NODE_DEF_OPEN_SCOPE

// Storage for caching GPU buffers and inference results
struct BssDeducerStorage {
    cuda::CUDALinearBufferHandle vertices_buffer;
    bool initialized = false;
    int num_vertices = 0;

    // Cache for inference results
    std::shared_ptr<ReducedOrderedBasis> cached_reduced_basis;
    std::vector<glm::vec3> cached_vertices;
    float cached_shape_code = -1.0f;
    bool cached_has_shape_code_2 = false;
    float cached_shape_code_2 = -1.0f;
    int cached_eigenfunction_idx = -1;
    int cached_eigenfunction_count = -1;
    std::string cached_model_path;
    bool cache_valid = false;

    constexpr static bool has_storage = false;
};

NODE_DECLARATION_FUNCTION(bss_deducer)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<float>("Shape Code").default_val(0.0f).min(0.0f).max(1.0f);
    b.add_input<bool>("Has Shape Code 2").default_val(false);
    b.add_input<float>("Shape Code 2").default_val(0.0f).min(0.0f).max(1.0f);
    b.add_input<int>("Eigenfunction Count").default_val(4).min(1).max(12);
    b.add_input<int>("Eigenfunction Index").default_val(0).min(0).max(3);
    b.add_input<std::string>("Result Name").default_val("eigenfunction");
    b.add_input<std::string>("Model Path")
        .default_val("mesh_deform_dirichlet_20260120_113410");
    b.add_output<Geometry>("Geometry");
    b.add_output<std::shared_ptr<ReducedOrderedBasis>>("Reduced Basis");
}

NODE_EXECUTION_FUNCTION(bss_deducer)
{
    spdlog::info("[BssDeducer] Node execution started");

    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();
    auto& storage = params.get_storage<BssDeducerStorage&>();

    float shape_code = params.get_input<float>("Shape Code");
    bool has_shape_code_2 = params.get_input<bool>("Has Shape Code 2");
    float shape_code_2 = params.get_input<float>("Shape Code 2");
    int eigenfunction_idx = params.get_input<int>("Eigenfunction Index");
    int eigenfunction_count = params.get_input<int>("Eigenfunction Count");
    std::string result_name = params.get_input<std::string>("Result Name");
    std::string model_path = params.get_input<std::string>("Model Path");

    // Get vertices from geometry
    auto mesh_component = input_geom.get_component<MeshComponent>();
    if (!mesh_component) {
        spdlog::warn("No mesh component found in geometry");
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        params.set_output<std::shared_ptr<ReducedOrderedBasis>>(
            "Reduced Basis", std::make_shared<ReducedOrderedBasis>());
        return true;
    }

    std::vector<glm::vec3> vertices = mesh_component->get_vertices();

    // Check if we can use cached results
    bool inputs_changed =
        !storage.cache_valid || storage.cached_shape_code != shape_code ||
        storage.cached_has_shape_code_2 != has_shape_code_2 ||
        storage.cached_shape_code_2 != shape_code_2 ||
        storage.cached_vertices.size() != vertices.size() ||
        storage.cached_vertices != vertices ||
        storage.cached_eigenfunction_count != eigenfunction_count ||
        storage.cached_model_path != model_path;

    if (!inputs_changed) {
        spdlog::info(
            "[BssDeducer] Using cached inference results (shape_code={:.2f})",
            shape_code);

        // Deep copy the cached reduced basis
        auto reduced_basis = std::make_shared<ReducedOrderedBasis>(
            *storage.cached_reduced_basis);

        // Add the selected eigenfunction as vertex scalar quantity
        if (eigenfunction_idx >= 0 &&
            eigenfunction_idx < reduced_basis->basis.size()) {
            std::vector<float> selected_results(
                reduced_basis->basis[eigenfunction_idx].size());
            Eigen::VectorXf::Map(
                selected_results.data(), selected_results.size()) =
                reduced_basis->basis[eigenfunction_idx];
            mesh_component->add_vertex_scalar_quantity(
                result_name, selected_results);
            spdlog::info(
                "Added cached eigenfunction {} as '{}' vertex attribute",
                eigenfunction_idx,
                result_name);
        }

        params.set_output<Geometry>("Geometry", std::move(input_geom));
        params.set_output<std::shared_ptr<ReducedOrderedBasis>>(
            "Reduced Basis", std::move(reduced_basis));
        return true;
    }

    // Inputs changed, need to recompute
    spdlog::info("[BssDeducer] Inputs changed, running inference...");

    auto reduced_basis = std::make_shared<ReducedOrderedBasis>();

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

        // Initialize basis set (will load model on first call)
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
            "Running inference on {} vertices with shape_code={:.2f}",
            vertices.size(),
            shape_code);

        // Zero-copy send CUDA array to Python
        float* gpu_ptr = storage.vertices_buffer->get_device_ptr<float>();
        python::send(
            "vertices_cuda",
            python::CudaArrayView<float>{ gpu_ptr, { vertices.size(), 3 } });

        // Send shape code to Python
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

        // Inference eigenfunctions based on eigenfunction_count
        reduced_basis->basis.resize(eigenfunction_count);
        std::vector<float> selected_results;

        for (int i = 0; i < eigenfunction_count; i++) {
            // Call Python inference function for each eigenfunction
            std::string inference_code =
                "import numpy as np\n"
                "import torch\n";

            if (has_shape_code_2) {
                inference_code +=
                    "inference_result_" + std::to_string(i) +
                    " = deducer.run_inference_on_vertices("
                    "vertices_cuda, eigenfunction_idx=" +
                    std::to_string(i) +
                    ", shape_code=torch.tensor(shape_code_values, "
                    "device='cuda'))";
            }
            else {
                inference_code +=
                    "inference_result_" + std::to_string(i) +
                    " = deducer.run_inference_on_vertices("
                    "vertices_cuda, eigenfunction_idx=" +
                    std::to_string(i) +
                    ", shape_code=torch.tensor([shape_code_value]))";
            }

            python::call<void>(inference_code);
            python::flush_python_output();  // Flush Python print output

            // Retrieve results as vector<float>
            std::vector<float> results = python::call<std::vector<float>>(
                "inference_result_" + std::to_string(i) + ".tolist()");

            if (results.size() != vertices.size()) {
                spdlog::error(
                    "Eigenfunction {} result size ({}) does not match vertex "
                    "count ({})",
                    i,
                    results.size(),
                    vertices.size());
                continue;
            }

            // Store in reduced basis
            reduced_basis->basis[i] =
                Eigen::VectorXf::Map(results.data(), results.size());

            // Keep the selected eigenfunction for quantity
            if (i == eigenfunction_idx) {
                selected_results = results;
            }

            spdlog::info("Computed eigenfunction {}", i);
        }

        // Update cache
        storage.cached_reduced_basis = reduced_basis;
        storage.cached_vertices = vertices;
        storage.cached_shape_code = shape_code;
        storage.cached_has_shape_code_2 = has_shape_code_2;
        storage.cached_shape_code_2 = shape_code_2;
        storage.cached_eigenfunction_idx = eigenfunction_idx;
        storage.cached_eigenfunction_count = eigenfunction_count;
        storage.cached_model_path = model_path;
        storage.cache_valid = true;
        spdlog::info("[BssDeducer] Cache updated (model={})", model_path);

        // Add only the selected eigenfunction as vertex scalar quantity
        if (!selected_results.empty()) {
            mesh_component->add_vertex_scalar_quantity(
                result_name, selected_results);
            spdlog::info(
                "Added eigenfunction {} as '{}' vertex attribute",
                eigenfunction_idx,
                result_name);
        }
    }
    catch (const std::exception& e) {
        spdlog::error("Python call failed: {}", e.what());
    }

    params.set_output<Geometry>("Geometry", std::move(input_geom));
    params.set_output<std::shared_ptr<ReducedOrderedBasis>>(
        "Reduced Basis", std::move(reduced_basis));

    return true;
}

NODE_DECLARATION_UI(bss_deducer);

NODE_DEF_CLOSE_SCOPE