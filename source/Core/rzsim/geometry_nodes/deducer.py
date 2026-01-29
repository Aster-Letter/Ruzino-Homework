import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add ShapeSpaceSpectra-reproduction project to Python path
project_root = Path(r"C:\Users\Pengfei\WorkSpace\ShapeSpaceSpectra-reproduction")
sys.path.insert(0, str(project_root))

from util.task_save_load import load_task_from_path

# Global basis_set to be loaded once
_basis_set = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_current_model_name = None


def initialize_basis_set(model_name):
    """
    Initialize and load the basis set model

    Args:
        model_name: Name of the model checkpoint folder (required, passed from C++)
    """
    global _basis_set, _device, _current_model_name

    # If already initialized with the same model, skip
    if _basis_set is not None and _current_model_name == model_name:
        return f"Basis set already initialized with model: {model_name}"

    # If different model requested, force re-initialization
    if _basis_set is not None and _current_model_name != model_name:
        _basis_set = None
        print(f"Switching from model '{_current_model_name}' to '{model_name}'")

    # Checkpoint path
    checkpoint_path = rf"C:\Users\Pengfei\WorkSpace\ShapeSpaceSpectra-reproduction\checkpoints\{model_name}"

    if not os.path.exists(checkpoint_path):
        return f"Error: Checkpoint path does not exist: {checkpoint_path}"

    # Change working directory to Ruzino root (where test.py runs successfully)
    ruzino_root = r"C:\Users\Pengfei\WorkSpace\Ruzino"
    original_cwd = os.getcwd()
    os.chdir(ruzino_root)

    try:
        # Load task from checkpoint folder
        task_path = os.path.join(checkpoint_path, "task")
        if not os.path.exists(task_path):
            return f"Error: Task folder not found in checkpoint: {task_path}"

        # Load basis set and configuration
        _basis_set, task_name, training_config = load_task_from_path(task_path)

        # Load network weights
        checkpoint_file = os.path.join(checkpoint_path, "network_params.pt")
        if not os.path.exists(checkpoint_file):
            return f"Error: Network parameters not found: {checkpoint_file}"

        _basis_set.load_weights(checkpoint_file, device=_device)
        _current_model_name = model_name

        return f"Initialized basis set: {task_name}, model={model_name}, {_basis_set.num_fields} eigenfunctions, device={_device}"
    except Exception as e:
        import traceback

        return f"Error initializing basis set: {str(e)}\n{traceback.format_exc()}"
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def generate_geometry_direct(shape_code_value=0.5):
    """
    Generate geometry directly using geometry_py API without saving to disk

    Args:
        shape_code_value: Shape code value(s) - can be single float or list of floats

    Returns:
        geometry_py.Geometry object or None on error
    """
    global _basis_set, _device

    if _basis_set is None:
        return None

    try:
        # Import geometry_py
        import geometry_py

        # Create shape code tensor
        if isinstance(shape_code_value, (list, tuple)):
            shape_code = torch.tensor(
                shape_code_value, device=_device, dtype=torch.float32
            )
        else:
            shape_code = torch.tensor(
                [shape_code_value], device=_device, dtype=torch.float32
            )

        print(f"[deducer] Generating geometry with shape_code={shape_code_value}")

        # Update shape code
        _basis_set._update_shape_code(shape_code)

        # Get geometry from shape_space
        geom = _basis_set.shape_space.geometry

        # Get vertices and faces (keep on GPU)
        vertices_torch = geom.vertices if hasattr(geom, "vertices") else geom.vertices
        faces_torch = geom.faces if hasattr(geom, "faces") else geom.faces

        num_vertices = vertices_torch.shape[0]
        num_faces = faces_torch.shape[0]

        print(f"[deducer] Mesh info: {num_vertices} vertices, {num_faces} faces")

        # Create empty geometry
        geometry = geometry_py.CreateMesh()
        mesh = geometry.get_mesh_component()
        cuda_view = mesh.get_cuda_view()

        # Set vertices directly from CUDA tensor (zero-copy or GPU-to-GPU copy)
        if vertices_torch.is_cuda:
            cuda_view.set_vertices(vertices_torch)
        else:
            cuda_view.set_vertices(vertices_torch.cuda())

        # Build face topology (assuming triangular faces)
        if faces_torch.shape[1] == 3:
            print(f"[deducer] Setting triangular face topology using CUDA tensors")
            # All triangles
            face_vertex_counts = torch.full(
                (num_faces,), 3, dtype=torch.int32, device=_device
            )
            face_vertex_indices = faces_torch.flatten().to(torch.int32)

            if face_vertex_counts.is_cuda:
                cuda_view.set_face_vertex_counts(face_vertex_counts)
            else:
                cuda_view.set_face_vertex_counts(face_vertex_counts.cuda())

            if face_vertex_indices.is_cuda:
                cuda_view.set_face_vertex_indices(face_vertex_indices)
            else:
                cuda_view.set_face_vertex_indices(face_vertex_indices.cuda())
        else:
            # Mixed face sizes - need to convert to CPU
            print(f"[deducer] Setting mixed face topology, transferring to CPU")
            faces_cpu = faces_torch.cpu().numpy()
            face_vertex_counts = [len(face) for face in faces_cpu]
            face_vertex_indices = []
            for face in faces_cpu:
                face_vertex_indices.extend([int(idx) for idx in face])

            mesh.set_face_vertex_counts(np.array(face_vertex_counts, dtype=np.int32))
            mesh.set_face_vertex_indices(np.array(face_vertex_indices, dtype=np.int32))

        print(f"[deducer] Successfully created geometry_py object using CUDA view")

        return geometry

    except Exception as e:
        print(f"[deducer] Error generating geometry: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def detect_boundary_from_geometry(
    geometry, shape_code_value=None, distance_threshold=1e-6
):
    """
    Detect Dirichlet boundary directly from geometry_py.Geometry object

    Args:
        geometry: geometry_py.Geometry object
        shape_code_value: Shape code value(s) - can be single float or list of floats
        distance_threshold: Distance threshold for boundary detection

    Returns:
        NumPy array with boundary values (1.0 for boundary, 0.0 for interior)
    """
    global _basis_set, _device

    if _basis_set is None:
        raise RuntimeError(
            "basis_set not initialized. Call initialize_basis_set first."
        )

    try:
        # Get mesh component and CUDA view
        mesh = geometry.get_mesh_component()
        cuda_view = mesh.get_cuda_view()

        # Get vertices as PyTorch CUDA tensor (zero-copy)
        vertices_cuda = cuda_view.get_vertices()

        # Ensure shape is (N, 3)
        if vertices_cuda.ndim == 1:
            vertices_cuda = vertices_cuda.reshape(-1, 3)

        # Prepare shape code
        if shape_code_value is None:
            shape_code = torch.tensor([0.0], device=_device)
        elif isinstance(shape_code_value, (list, tuple)):
            shape_code = torch.tensor(shape_code_value, device=_device)
        else:
            shape_code = torch.tensor([float(shape_code_value)], device=_device)

        # Call boundary detection
        with torch.no_grad():
            boundary_result = _basis_set.is_on_dirichlet_boundary(
                shape_code=shape_code,
                sample_points=vertices_cuda,
                distance_threshold=distance_threshold,
            )

        # Convert to numpy
        if boundary_result.dtype == torch.bool:
            boundary_values = boundary_result.float()
        else:
            boundary_values = boundary_result

        cuda_view.add_vertex_scalar_quantity("nn_dirichlet", boundary_values)

        return

    except Exception as e:
        print(f"[deducer] Error in detect_boundary_from_geometry: {str(e)}")
        import traceback

        traceback.print_exc()
        return np.array([])


def run_inference_from_geometry(geometry, eigenfunction_idx=0, shape_code_value=None):
    """
    Run inference directly from geometry_py.Geometry object

    Args:
        geometry: geometry_py.Geometry object
        eigenfunction_idx: Index of eigenfunction to compute
        shape_code_value: Shape code value(s) - can be single float or list of floats

    Returns:
        NumPy array with inference results
    """
    global _basis_set, _device

    if _basis_set is None:
        raise RuntimeError(
            "basis_set not initialized. Call initialize_basis_set first."
        )

    try:
        # Get mesh component and CUDA view
        mesh = geometry.get_mesh_component()
        cuda_view = mesh.get_cuda_view()

        # Get vertices as PyTorch CUDA tensor (zero-copy)
        vertices_cuda = cuda_view.get_vertices()

        # Ensure shape is (N, 3)
        if vertices_cuda.ndim == 1:
            vertices_cuda = vertices_cuda.reshape(-1, 3)

        num_vertices = vertices_cuda.shape[0]

        # Prepare shape code
        if shape_code_value is None:
            shape_code = torch.tensor([0.0], device=_device)
        elif isinstance(shape_code_value, (list, tuple)):
            shape_code = torch.tensor(shape_code_value, device=_device)
        else:
            shape_code = torch.tensor([float(shape_code_value)], device=_device)

        # Clamp eigenfunction index
        eigenfunction_idx = max(0, min(eigenfunction_idx, _basis_set.num_fields - 1))

        # Run inference
        with torch.no_grad():
            output = _basis_set.inference(
                i=eigenfunction_idx,
                shape_code=shape_code,
                sample=vertices_cuda,
                device=_device,
            )

        # Convert to numpy
        output_cpu = output.squeeze().cpu().numpy()

        return output_cpu

    except Exception as e:
        print(f"[deducer] Error in run_inference_from_geometry: {str(e)}")
        import traceback

        traceback.print_exc()
        return np.array([])
