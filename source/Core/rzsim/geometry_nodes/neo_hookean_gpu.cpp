#include <RHI/cuda.hpp>
#include <fstream>
#include <glm/glm.hpp>
#include <limits>

#include "GCore/Components/MeshComponent.h"
#include "GCore/Components/PointsComponent.h"
#include "GCore/geom_payload.hpp"
#include "RHI/internal/cuda_extension.hpp"
#include "RZSolver/Solver.hpp"
#include "glm/ext/vector_float3.hpp"
#include "nodes/core/def/node_def.hpp"
#include "rzsim_cuda/adjacency_map.cuh"
#include "rzsim_cuda/mass_spring_implicit.cuh"
#include "rzsim_cuda/neo_hookean.cuh"
#include "spdlog/spdlog.h"

NODE_DEF_OPEN_SCOPE

// Storage for persistent GPU simulation state
struct NeoHookeanGPUStorage {
    cuda::CUDALinearBufferHandle positions_buffer;
    cuda::CUDALinearBufferHandle velocities_buffer;
    cuda::CUDALinearBufferHandle Dm_inv_buffer;
    cuda::CUDALinearBufferHandle volumes_buffer;
    cuda::CUDALinearBufferHandle element_to_vertex_buffer;
    cuda::CUDALinearBufferHandle element_to_local_face_buffer;
    cuda::CUDALinearBufferHandle next_positions_buffer;
    cuda::CUDALinearBufferHandle mass_matrix_buffer;
    cuda::CUDALinearBufferHandle gradients_buffer;
    cuda::CUDALinearBufferHandle f_ext_buffer;

    // Mesh topology buffers (cached)
    cuda::CUDALinearBufferHandle face_vertex_indices_buffer;
    cuda::CUDALinearBufferHandle face_counts_buffer;
    cuda::CUDALinearBufferHandle normals_buffer;
    cuda::CUDALinearBufferHandle adjacency_buffer;
    cuda::CUDALinearBufferHandle offsets_buffer;

    // Pre-built CSR structure (built once, reused forever)
    rzsim_cuda::NeoHookeanCSRStructure hessian_structure;
    cuda::CUDALinearBufferHandle hessian_values;

    // Temporary buffers for Newton iterations
    cuda::CUDALinearBufferHandle x_new_buffer;
    cuda::CUDALinearBufferHandle newton_direction_buffer;
    cuda::CUDALinearBufferHandle neg_gradient_buffer;
    cuda::CUDALinearBufferHandle x_candidate_buffer;

    // Temporary buffers for energy computation
    cuda::CUDALinearBufferHandle inertial_terms_buffer;
    cuda::CUDALinearBufferHandle element_energies_buffer;
    cuda::CUDALinearBufferHandle potential_terms_buffer;

    // Reuse solver instance
    std::unique_ptr<Ruzino::Solver::LinearSolver> solver;

    bool initialized = false;
    int num_particles = 0;
    int num_elements = 0;

    constexpr static bool has_storage = false;

    // Initialize all GPU buffers and structures
    void initialize(
        const std::vector<glm::vec3>& positions,
        const std::vector<int>& face_vertex_indices,
        const std::vector<int>& face_counts,
        float density)
    {
        num_particles = positions.size();

        // Write positions to GPU buffer
        positions_buffer = cuda::create_cuda_linear_buffer(positions);
        // Cache face topology buffers
        face_vertex_indices_buffer =
            cuda::create_cuda_linear_buffer(face_vertex_indices);
        face_counts_buffer = cuda::create_cuda_linear_buffer(face_counts);

        // Compute volume adjacency (tetrahedra reconstruction)
        unsigned num_elements_gpu;
        std::tie(adjacency_buffer, offsets_buffer, num_elements_gpu) =
            rzsim_cuda::compute_volume_adjacency_gpu(
                positions_buffer, face_vertex_indices_buffer);

        num_elements = num_elements_gpu;

        if (num_elements == 0) {
            spdlog::error(
                "No tetrahedral elements found! Neo-Hookean requires "
                "volumetric mesh.");
            return;
        }

        // Initialize velocities to zero
        std::vector<glm::vec3> initial_velocities(
            num_particles, glm::vec3(0.0f));
        velocities_buffer = cuda::create_cuda_linear_buffer(initial_velocities);

        next_positions_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        gradients_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        f_ext_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Mass matrix will be computed from density and volumes after reference data is ready
        mass_matrix_buffer = cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Compute reference shape matrices and volumes
        auto [Dm_inv, volumes, element_to_vertex, element_to_local_face] =
            rzsim_cuda::compute_reference_data_gpu(
                positions_buffer,
                adjacency_buffer,
                offsets_buffer,
                num_elements);

        Dm_inv_buffer = Dm_inv;
        volumes_buffer = volumes;
        element_to_vertex_buffer = element_to_vertex;
        element_to_local_face_buffer = element_to_local_face;

        // Compute lumped mass matrix from density and element volumes
        // For each element: m_elem = density * volume
        // Distribute equally to 4 vertices: m_vertex += m_elem / 4
        rzsim_cuda::compute_lumped_mass_matrix_gpu(
            adjacency_buffer,
            offsets_buffer,
            element_to_vertex_buffer,
            element_to_local_face_buffer,
            volumes_buffer,
            density,
            num_particles,
            num_elements,
            mass_matrix_buffer);

        // Build Hessian CSR structure
        hessian_structure = rzsim_cuda::build_hessian_structure_nh_gpu(
            adjacency_buffer,
            offsets_buffer,
            element_to_vertex_buffer,
            element_to_local_face_buffer,
            num_particles,
            num_elements);

        hessian_values =
            cuda::create_cuda_linear_buffer<float>(hessian_structure.nnz);

        // Allocate temporary buffers for Newton iterations
        x_new_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        newton_direction_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        neg_gradient_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        x_candidate_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Allocate temporary buffers for energy computation
        inertial_terms_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);
        element_energies_buffer =
            cuda::create_cuda_linear_buffer<float>(num_elements);
        potential_terms_buffer =
            cuda::create_cuda_linear_buffer<float>(num_particles * 3);

        // Create solver instance
        solver = Ruzino::Solver::SolverFactory::create(
            Ruzino::Solver::SolverType::CUDA_CG);

        initialized = true;
    }
};

NODE_DECLARATION_FUNCTION(neo_hookean_gpu)
{
    b.add_input<Geometry>("Geometry");
    b.add_input<float>("Density").default_val(1000.0f).min(1.0f).max(10000.0f);
    b.add_input<float>("Young's Modulus").default_val(5e4f).min(1e3f).max(1e9f);
    b.add_input<float>("Poisson's Ratio")
        .default_val(0.35f)
        .min(0.0f)
        .max(0.49f);
    b.add_input<float>("Damping").default_val(0.99f).min(0.0f).max(1.0f);
    b.add_input<int>("Substeps").default_val(5).min(1).max(20);
    b.add_input<int>("Newton Iterations").default_val(50).min(1).max(100);
    b.add_input<float>("Newton Tolerance")
        .default_val(1e-2f)
        .min(1e-8f)
        .max(1e-1f);
    b.add_input<float>("Gravity").default_val(-9.81f).min(-20.0f).max(0.0f);
    b.add_input<float>("Ground Restitution")
        .default_val(0.3f)
        .min(0.0f)
        .max(1.0f);
    b.add_input<bool>("Flip Normal").default_val(false);

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(neo_hookean_gpu)
{
    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto& storage = params.get_storage<NeoHookeanGPUStorage&>();

    // Open log file for debugging
    static std::ofstream debug_log("neo_hookean_debug.log", std::ios::out);
    static int frame_count = 0;
    frame_count++;

    // Get inputs
    auto input_geom = params.get_input<Geometry>("Geometry");
    input_geom.apply_transform();

    float density = params.get_input<float>("Density");
    float youngs_modulus = params.get_input<float>("Young's Modulus");
    float poisson_ratio = params.get_input<float>("Poisson's Ratio");
    float damping = params.get_input<float>("Damping");
    int substeps = params.get_input<int>("Substeps");
    int max_iterations = params.get_input<int>("Newton Iterations");
    float tolerance = params.get_input<float>("Newton Tolerance");
    tolerance = std::max(tolerance, 1e-8f);
    float gravity = params.get_input<float>("Gravity");
    float restitution = params.get_input<float>("Ground Restitution");
    bool flip_normal = params.get_input<bool>("Flip Normal");
    float dt = global_payload.delta_time;

    // Convert Young's modulus and Poisson's ratio to Lamé parameters
    float mu = youngs_modulus / (2.0f * (1.0f + poisson_ratio));
    float lambda = youngs_modulus * poisson_ratio /
                   ((1.0f + poisson_ratio) * (1.0f - 2.0f * poisson_ratio));

    // Get mesh component
    auto mesh_component = input_geom.get_component<MeshComponent>();
    std::vector<glm::vec3> positions;
    std::vector<int> face_vertex_indices;
    std::vector<int> face_counts;

    if (mesh_component) {
        positions = mesh_component->get_vertices();
        face_vertex_indices = mesh_component->get_face_vertex_indices();
        face_counts = mesh_component->get_face_vertex_counts();
    }
    else {
        auto points_component = input_geom.get_component<PointsComponent>();
        if (!points_component) {
            params.set_output<Geometry>("Geometry", std::move(input_geom));
            return true;
        }
        positions = points_component->get_vertices();
    }

    int num_particles = positions.size();
    if (num_particles == 0) {
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    // Initialize buffers only once or when particle count changes
    if (!storage.initialized || storage.num_particles != num_particles) {
        storage.initialize(positions, face_vertex_indices, face_counts, density);
    }

    if (!storage.initialized || storage.num_elements == 0) {
        spdlog::warn(
            "[NeoHookean] Neo-Hookean simulation requires tetrahedral mesh. "
            "Skipping simulation.");
        params.set_output<Geometry>("Geometry", std::move(input_geom));
        return true;
    }

    auto d_positions = storage.positions_buffer;
    auto d_velocities = storage.velocities_buffer;
    auto d_next_positions = storage.next_positions_buffer;
    auto d_M_diag = storage.mass_matrix_buffer;
    auto d_gradients = storage.gradients_buffer;
    auto d_f_ext = storage.f_ext_buffer;

    // Log initial state
    if (frame_count == 1) {
        auto initial_pos = d_positions->get_host_vector<glm::vec3>();
        auto initial_vel = d_velocities->get_host_vector<glm::vec3>();
        debug_log << "\n===== INITIAL STATE (Frame " << frame_count << ") =====\n";
        debug_log << "Number of particles: " << num_particles << "\n";
        debug_log << "Number of elements: " << storage.num_elements << "\n";
        debug_log << "dt = " << dt << ", substeps = " << substeps << ", dt_sub = " << (dt/substeps) << "\n";
        debug_log << "mu = " << mu << ", lambda = " << lambda << "\n";
        debug_log << "density = " << density << ", gravity = " << gravity << "\n\n";
        
        debug_log << "Initial positions:\n";
        for (int i = 0; i < num_particles; i++) {
            debug_log << "  Vertex " << i << ": [" 
                     << initial_pos[i].x << ", " 
                     << initial_pos[i].y << ", " 
                     << initial_pos[i].z << "]\n";
        }
        
        debug_log << "\nInitial velocities:\n";
        for (int i = 0; i < num_particles; i++) {
            debug_log << "  Vertex " << i << ": [" 
                     << initial_vel[i].x << ", " 
                     << initial_vel[i].y << ", " 
                     << initial_vel[i].z << "]\n";
        }
        
        // Log reference data (Dm_inv and volumes)
        auto Dm_inv_host = storage.Dm_inv_buffer->get_host_vector<float>();
        auto volumes_host = storage.volumes_buffer->get_host_vector<float>();
        auto elem_to_v = storage.element_to_vertex_buffer->get_host_vector<int>();
        auto elem_to_lf = storage.element_to_local_face_buffer->get_host_vector<int>();
        auto mass_host = storage.mass_matrix_buffer->get_host_vector<float>();
        
        debug_log << "\nMass matrix (per vertex, per DOF):\n";
        for (int i = 0; i < num_particles; i++) {
            debug_log << "  Vertex " << i << ": m = [" 
                     << mass_host[i*3+0] << ", "
                     << mass_host[i*3+1] << ", "
                     << mass_host[i*3+2] << "]\n";
        }
        
        debug_log << "\nReference data (Dm_inv and volumes):\n";
        for (int e = 0; e < storage.num_elements; e++) {
            debug_log << "  Element " << e << ": volume = " << volumes_host[e] << "\n";
            debug_log << "    apex vertex = " << elem_to_v[e] 
                     << ", local_face = " << elem_to_lf[e] << "\n";
            debug_log << "    Dm_inv =\n";
            for (int i = 0; i < 3; i++) {
                debug_log << "      [" 
                         << Dm_inv_host[e*9 + i*3 + 0] << ", "
                         << Dm_inv_host[e*9 + i*3 + 1] << ", "
                         << Dm_inv_host[e*9 + i*3 + 2] << "]\n";
            }
        }
        
        // Log external forces
        auto f_ext_host = d_f_ext->get_host_vector<float>();
        debug_log << "\nExternal forces (from FEM integration):\n";
        for (int i = 0; i < num_particles; i++) {
            debug_log << "  Vertex " << i << ": f_ext = [" 
                     << f_ext_host[i*3+0] << ", "
                     << f_ext_host[i*3+1] << ", "
                     << f_ext_host[i*3+2] << "]\n";
        }
        
        debug_log.flush();
    }

    // Substep loop
    float dt_sub = dt / substeps;

    for (int substep = 0; substep < substeps; ++substep) {
        // Setup external forces on GPU (using proper FEM integration)
        rzsim_cuda::setup_external_forces_fem_gpu(
            storage.adjacency_buffer,
            storage.offsets_buffer,
            storage.element_to_vertex_buffer,
            storage.element_to_local_face_buffer,
            storage.volumes_buffer,
            density,
            gravity,
            num_particles,
            storage.num_elements,
            d_f_ext);

        // Compute x_tilde = x + dt_sub * v on GPU
        rzsim_cuda::explicit_step_nh_gpu(
            d_positions, d_velocities, dt_sub, num_particles, d_next_positions);

        // Newton's method iterations
        storage.x_new_buffer->copy_from_device(d_next_positions.Get());

        bool converged = false;

        for (int iter = 0; iter < max_iterations; iter++) {
            // Compute gradient at current x_new
            rzsim_cuda::compute_gradient_nh_gpu(
                storage.x_new_buffer,
                d_next_positions,
                d_M_diag,
                d_f_ext,
                storage.adjacency_buffer,
                storage.offsets_buffer,
                storage.element_to_vertex_buffer,
                storage.element_to_local_face_buffer,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                dt_sub,
                num_particles,
                storage.num_elements,
                d_gradients);

            cudaError_t err = cudaGetLastError();

            float grad_norm = rzsim_cuda::compute_vector_norm_nh_gpu(
                d_gradients, num_particles * 3);

            // Detailed logging for first substep and first few iterations
            if (substep == 0 && iter < 3) {
                auto grad_host = d_gradients->get_host_vector<float>();
                auto x_curr_host = storage.x_new_buffer->get_host_vector<float>();
                auto x_tilde_host = d_next_positions->get_host_vector<float>();
                
                debug_log << "\n========== Frame " << frame_count << ", Substep " << substep 
                         << ", Iteration " << iter << " ==========\n";
                debug_log << "Gradient (all " << num_particles * 3 << " DOFs):\n";
                for (int i = 0; i < num_particles; i++) {
                    debug_log << "  Vertex " << i << ": grad = [" 
                             << grad_host[i*3+0] << ", " 
                             << grad_host[i*3+1] << ", " 
                             << grad_host[i*3+2] << "]\n";
                }
                
                debug_log << "\nCurrent positions (x_new):\n";
                for (int i = 0; i < num_particles; i++) {
                    debug_log << "  Vertex " << i << ": x = [" 
                             << x_curr_host[i*3+0] << ", " 
                             << x_curr_host[i*3+1] << ", " 
                             << x_curr_host[i*3+2] << "]\n";
                }
                
                debug_log << "\nPredicted positions (x_tilde = x + dt*v):\n";
                for (int i = 0; i < num_particles; i++) {
                    debug_log << "  Vertex " << i << ": x_tilde = [" 
                             << x_tilde_host[i*3+0] << ", " 
                             << x_tilde_host[i*3+1] << ", " 
                             << x_tilde_host[i*3+2] << "]\n";
                }
                
                debug_log << "\nDisplacement (x_new - x_tilde):\n";
                for (int i = 0; i < num_particles; i++) {
                    debug_log << "  Vertex " << i << ": dx = [" 
                             << (x_curr_host[i*3+0] - x_tilde_host[i*3+0]) << ", " 
                             << (x_curr_host[i*3+1] - x_tilde_host[i*3+1]) << ", " 
                             << (x_curr_host[i*3+2] - x_tilde_host[i*3+2]) << "]\n";
                }
                
                debug_log << "\nGradient norm: " << grad_norm << "\n";
                debug_log.flush();
            }

            spdlog::info("[NeoHookean] Gradient norm: {}", grad_norm);

            if (!std::isfinite(grad_norm)) {
                spdlog::error(
                    "[NeoHookean] Gradient norm is not finite! Simulation "
                    "unstable.");
                break;
            }

            auto dof = num_particles * 3;
            grad_norm = grad_norm / dof;

            if (iter > 0 && grad_norm < tolerance) {
                converged = true;
                break;
            }

            // Update Hessian values
            rzsim_cuda::update_hessian_values_nh_gpu(
                storage.hessian_structure,
                storage.x_new_buffer,
                d_M_diag,
                storage.adjacency_buffer,
                storage.offsets_buffer,
                storage.element_to_vertex_buffer,
                storage.element_to_local_face_buffer,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                dt_sub,
                num_particles,
                storage.num_elements,
                storage.hessian_values);

            // DEBUG: Export dense Hessian (disabled for performance)
            constexpr bool enable_hessian_export = false;
            if (enable_hessian_export && substep == 0 && iter == 0) {
                spdlog::info(
                    "[NeoHookean] !!! Exporting dense Hessian to CSV...");

                auto hess_vals =
                    storage.hessian_values->get_host_vector<float>();
                auto row_offsets = storage.hessian_structure.row_offsets
                                       ->get_host_vector<int>();
                auto col_indices = storage.hessian_structure.col_indices
                                       ->get_host_vector<int>();
                int n = storage.hessian_structure.num_rows;

                spdlog::info(
                    "[NeoHookean] Matrix size: {}x{}, nnz={}",
                    n,
                    n,
                    storage.hessian_structure.nnz);

                // Convert CSR to dense
                std::vector<std::vector<float>> dense(
                    n, std::vector<float>(n, 0.0f));
                for (int i = 0; i < n; i++) {
                    int row_start = row_offsets[i];
                    int row_end = row_offsets[i + 1];
                    for (int j = row_start; j < row_end; j++) {
                        int col = col_indices[j];
                        dense[i][col] = hess_vals[j];
                    }
                }

                spdlog::info("[NeoHookean] Writing to CSV file...");
                // Write to CSV
                std::ofstream csv("hessian_debug.csv");
                if (!csv.is_open()) {
                    spdlog::error(
                        "[NeoHookean] Failed to open hessian_debug.csv for "
                        "writing!");
                }
                else {
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n; j++) {
                            csv << dense[i][j];
                            if (j < n - 1)
                                csv << ",";
                        }
                        csv << "\n";
                    }
                    csv.close();
                    spdlog::info(
                        "[NeoHookean] !!! Dense Hessian written to "
                        "hessian_debug.csv ({}x{})",
                        n,
                        n);
                }

                // Write to Mathematica format
                spdlog::info("[NeoHookean] Writing to Mathematica format...");
                std::ofstream mma("hessian_debug.m");
                if (!mma.is_open()) {
                    spdlog::error(
                        "[NeoHookean] Failed to open hessian_debug.m for "
                        "writing!");
                }
                else {
                    mma << "{";
                    for (int i = 0; i < n; i++) {
                        mma << "{";
                        for (int j = 0; j < n; j++) {
                            mma << dense[i][j];
                            if (j < n - 1)
                                mma << ", ";
                        }
                        mma << "}";
                        if (i < n - 1)
                            mma << ",\n";
                    }
                    mma << "}";
                    mma.close();
                    spdlog::info(
                        "[NeoHookean] !!! Dense Hessian written to "
                        "hessian_debug.m ({}x{})",
                        n,
                        n);
                }
            }

            // Log Hessian diagonal for debugging
            if (substep == 0 && iter < 3) {
                auto hess_vals_debug = storage.hessian_values->get_host_vector<float>();
                auto hess_row_offsets = storage.hessian_structure.row_offsets->get_host_vector<int>();
                auto hess_cols = storage.hessian_structure.col_indices->get_host_vector<int>();
                
                debug_log << "\nHessian diagonal:\n";
                for (int i = 0; i < num_particles * 3; i++) {
                    int row_start = hess_row_offsets[i];
                    int row_end = hess_row_offsets[i + 1];
                    for (int j = row_start; j < row_end; j++) {
                        if (hess_cols[j] == i) {
                            debug_log << "  H[" << i << "," << i << "] = " 
                                     << hess_vals_debug[j] << "\n";
                            break;
                        }
                    }
                }
                debug_log.flush();
            }

            // Debug: Check elastic contribution to Hessian
            // For initial state with no deformation, elastic contribution
            // should be near zero
            auto M_diag_host =
                storage.mass_matrix_buffer->get_host_vector<float>();
            auto hess_vals_debug =
                storage.hessian_values->get_host_vector<float>();
            auto mass_positions_debug =
                storage.hessian_structure.mass_value_positions
                    ->get_host_vector<int>();

            // Compare mass diagonal vs total diagonal
            spdlog::info(
                "[NeoHookean] DEBUG: Checking elastic contribution to Hessian");
            float max_elastic_contribution = 0.0f;
            auto hess_row_offsets =
                storage.hessian_structure.row_offsets->get_host_vector<int>();
            auto hess_col_indices =
                storage.hessian_structure.col_indices->get_host_vector<int>();

            for (int i = 0; i < num_particles * 3; i++) {
                int mass_pos = mass_positions_debug[i];
                float mass_val = M_diag_host[i];
                float total_diag = 0.0f;

                // Find diagonal value in CSR
                int row_start = hess_row_offsets[i];
                int row_end = hess_row_offsets[i + 1];
                for (int j = row_start; j < row_end; j++) {
                    if (hess_col_indices[j] == i) {
                        total_diag = hess_vals_debug[j];
                        break;
                    }
                }

                float elastic_contrib = total_diag - mass_val;
                max_elastic_contribution = std::max(
                    max_elastic_contribution, std::abs(elastic_contrib));

                if (i < 3) {
                    spdlog::info(
                        "  DOF {}: M={:.6f}, H_diag={:.6f}, elastic={:.6f}",
                        i,
                        mass_val,
                        total_diag,
                        elastic_contrib);
                }
            }

            spdlog::info(
                "[NeoHookean] Max elastic contribution to diagonal: {:.6e}",
                max_elastic_contribution);
            spdlog::info(
                "[NeoHookean] Expected M/dt^2 = {:.6f}",
                M_diag_host[0] / (dt_sub * dt_sub));

            // If elastic contribution is negligible, the problem is purely
            // quadratic and should converge in 1 Newton iteration
            if (max_elastic_contribution < M_diag_host[0] * 0.01f) {
                spdlog::info(
                    "[NeoHookean] Elastic contribution negligible - problem is "
                    "purely quadratic!");
            }

            spdlog::info(
                "[NeoHookean] Preparing to solve linear system for Newton "
                "direction...");

            // Solve H * p = -grad using CUDA CG
            // Use more relaxed tolerance since we're in an outer Newton loop
            float cg_tol = std::max(1e-6f, grad_norm * 0.1f);

            Ruzino::Solver::SolverConfig solver_config;
            solver_config.tolerance = cg_tol;
            solver_config.max_iterations = 500;  // Increased from 1000
            solver_config.use_preconditioner = true;
            solver_config.verbose = false;

            // Negate gradient for RHS
            rzsim_cuda::negate_nh_gpu(
                d_gradients, storage.neg_gradient_buffer, num_particles * 3);

            // Zero out the solution buffer before solving
            cudaMemset(
                reinterpret_cast<void*>(
                    storage.newton_direction_buffer->get_device_ptr()),
                0,
                num_particles * 3 * sizeof(float));

            // Solve on GPU
            auto result = storage.solver->solveGPU(
                storage.hessian_structure.num_rows,
                storage.hessian_structure.nnz,
                reinterpret_cast<const int*>(
                    storage.hessian_structure.row_offsets->get_device_ptr()),
                reinterpret_cast<const int*>(
                    storage.hessian_structure.col_indices->get_device_ptr()),
                reinterpret_cast<const float*>(
                    storage.hessian_values->get_device_ptr()),
                reinterpret_cast<const float*>(
                    storage.neg_gradient_buffer->get_device_ptr()),
                reinterpret_cast<float*>(
                    storage.newton_direction_buffer->get_device_ptr()),
                solver_config);

            if (!result.converged) {
                spdlog::warn(
                    "[NeoHookean] CG solver did not converge in iteration {}",
                    iter);
            }

            // Log Newton direction
            if (substep == 0 && iter < 3) {
                auto newton_dir_host = storage.newton_direction_buffer->get_host_vector<float>();
                debug_log << "\nNewton direction (solution to H * d = -g):\n";
                for (int i = 0; i < num_particles; i++) {
                    debug_log << "  Vertex " << i << ": d = [" 
                             << newton_dir_host[i*3+0] << ", " 
                             << newton_dir_host[i*3+1] << ", " 
                             << newton_dir_host[i*3+2] << "]\n";
                }
                
                // Compute direction magnitude and orientation
                float total_dx = 0, total_dy = 0, total_dz = 0;
                for (int i = 0; i < num_particles; i++) {
                    total_dx += newton_dir_host[i*3+0];
                    total_dy += newton_dir_host[i*3+1];
                    total_dz += newton_dir_host[i*3+2];
                }
                debug_log << "\nTotal displacement: [" << total_dx << ", " 
                         << total_dy << ", " << total_dz << "]\n";
                debug_log << "Expected: vertical (z-direction) only\n";
                debug_log.flush();
            }

            // Line search with energy descent
            // IMPORTANT: Do NOT update x_new before line search!
            spdlog::info("[NeoHookean] Starting line search...");
            float E_current = rzsim_cuda::compute_energy_nh_gpu(
                storage.x_new_buffer,
                d_next_positions,
                d_M_diag,
                d_f_ext,
                storage.adjacency_buffer,
                storage.offsets_buffer,
                storage.element_to_vertex_buffer,
                storage.element_to_local_face_buffer,
                storage.Dm_inv_buffer,
                storage.volumes_buffer,
                mu,
                lambda,
                dt_sub,
                num_particles,
                storage.num_elements,
                storage.inertial_terms_buffer,
                storage.element_energies_buffer,
                storage.potential_terms_buffer);

            spdlog::info(
                "[NeoHookean] Current energy: {:.6f}, grad_norm: {:.6f}",
                E_current,
                grad_norm);
            
            // DEBUG: For first iteration, check if problem is quadratic
            if (substep == 0 && iter == 0) {
                spdlog::info("[NeoHookean] === QUADRATIC PROBLEM DEBUG ===");
                spdlog::info("[NeoHookean] First iteration should be exact for quadratic problem!");
                
                // Check gradient dot newton_direction (should be negative)
                auto grad_host = d_gradients->get_host_vector<float>();
                auto p_host = storage.newton_direction_buffer->get_host_vector<float>();
                float grad_dot_p = 0.0f;
                for (int i = 0; i < num_particles * 3; i++) {
                    grad_dot_p += grad_host[i] * p_host[i];
                }
                spdlog::info("[NeoHookean] grad · p = {:.6e} (should be NEGATIVE for descent)", grad_dot_p);
                
                // Compute predicted energy decrease (quadratic model)
                // E(x+p) ≈ E(x) + grad·p + 0.5*p·H·p
                // For Newton step: H*p = -grad, so p·H·p = -p·grad = -grad·p
                // Thus: E(x+p) ≈ E(x) + grad·p - 0.5*grad·p = E(x) + 0.5*grad·p
                float predicted_decrease = 0.5f * grad_dot_p;
                spdlog::info("[NeoHookean] Predicted energy decrease: {:.6e}", -predicted_decrease);
                spdlog::info("[NeoHookean] Predicted E(x+p): {:.6f}", E_current + predicted_decrease);
            }

            float E_candidate = std::numeric_limits<float>::infinity();
            float alpha = 1.0f;  // Start with full Newton step
            int ls_iter = 0;

            while (E_candidate > E_current && ls_iter < 200) {
                // x_candidate = x_new + alpha * p
                rzsim_cuda::axpy_nh_gpu(
                    alpha,
                    storage.newton_direction_buffer,
                    storage.x_new_buffer,
                    storage.x_candidate_buffer,
                    num_particles * 3);

                E_candidate = rzsim_cuda::compute_energy_nh_gpu(
                    storage.x_candidate_buffer,
                    d_next_positions,
                    d_M_diag,
                    d_f_ext,
                    storage.adjacency_buffer,
                    storage.offsets_buffer,
                    storage.element_to_vertex_buffer,
                    storage.element_to_local_face_buffer,
                    storage.Dm_inv_buffer,
                    storage.volumes_buffer,
                    mu,
                    lambda,
                    dt_sub,
                    num_particles,
                    storage.num_elements,
                    storage.inertial_terms_buffer,
                    storage.element_energies_buffer,
                    storage.potential_terms_buffer);


                bool accept = E_candidate <= E_current;
                if (accept) {
                    storage.x_new_buffer->copy_from_device(
                        storage.x_candidate_buffer.Get());
                    spdlog::info(
                        "[NeoHookean] Line search accepted at iter {}, "
                        "alpha={:.2e}, E_candidate={:.8f}",
                        ls_iter,
                        alpha,
                        E_candidate);
                    break;
                }

                alpha *= 0.5f;
                ls_iter++;
            }

            spdlog::info(
                "[NeoHookean] Line search ended: ls_iter={}, alpha={:.2e}",
                ls_iter,
                alpha);
            if (ls_iter >= 200 || alpha < 1e-6f) {
                spdlog::warn(
                    "Line search failed: ls_iter={}, alpha={:.2e}",
                    ls_iter,
                    alpha);
                // If line search fails completely, still take a tiny step to
                // make progress
                if (alpha < 1e-6f) {
                    spdlog::info("[NeoHookean] Forcing update with alpha=1e-6");
                    rzsim_cuda::axpy_nh_gpu(
                        1e-6f,
                        storage.newton_direction_buffer,
                        storage.x_new_buffer,
                        storage.x_candidate_buffer,
                        num_particles * 3);
                    storage.x_new_buffer->copy_from_device(
                        storage.x_candidate_buffer.Get());
                }
            }
        }

        // Check if Newton method converged
        if (!converged) {
            spdlog::error(
                "[NeoHookean] Newton method FAILED to converge after {} "
                "iterations!",
                max_iterations);
            spdlog::error(
                "[NeoHookean] This indicates a serious problem with the "
                "simulation.");
            // Don't break - let the simulation continue but warn the user
        }

        // Update velocities: v = (x_new - x_n) / dt_sub and apply damping
        auto x_new_final = storage.x_new_buffer->get_host_vector<float>();
        auto x_n_host = d_positions->get_host_vector<glm::vec3>();
        std::vector<glm::vec3> v_new(num_particles);
        for (int i = 0; i < num_particles; i++) {
            v_new[i].x =
                (x_new_final[i * 3 + 0] - x_n_host[i].x) / dt_sub * damping;
            v_new[i].y =
                (x_new_final[i * 3 + 1] - x_n_host[i].y) / dt_sub * damping;
            v_new[i].z =
                (x_new_final[i * 3 + 2] - x_n_host[i].z) / dt_sub * damping;
        }

        // Handle ground collision (z = 0)
        for (int i = 0; i < num_particles; i++) {
            float z_new = x_new_final[i * 3 + 2];
            if (z_new < 0.0f) {
                x_new_final[i * 3 + 2] = 0.0f;

                // Reflect velocity with restitution
                if (v_new[i].z < 0.0f) {
                    v_new[i].z = -v_new[i].z * restitution;
                }
            }
        }

        // Convert to output format
        std::vector<glm::vec3> new_positions(num_particles);
        for (int i = 0; i < num_particles; i++) {
            new_positions[i].x = x_new_final[i * 3 + 0];
            new_positions[i].y = x_new_final[i * 3 + 1];
            new_positions[i].z = x_new_final[i * 3 + 2];
        }

        d_velocities->assign_host_vector(v_new);
        d_positions->assign_host_vector(new_positions);
    }

    // Update geometry with new positions
    if (mesh_component) {
        auto final_positions = d_positions->get_host_vector<glm::vec3>();
        mesh_component->set_vertices(final_positions);

        // Note: For proper rendering, you'd want to recompute normals
        // For now, we'll keep the original normals or recompute them from
        // surface
    }
    else {
        auto points_component = input_geom.get_component<PointsComponent>();
        auto final_positions = d_positions->get_host_vector<glm::vec3>();
        points_component->set_vertices(final_positions);
    }

    params.set_output<Geometry>("Geometry", std::move(input_geom));
    return true;
}

NODE_DECLARATION_UI(neo_hookean_gpu);
NODE_DEF_CLOSE_SCOPE
