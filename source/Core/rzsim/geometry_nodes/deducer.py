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


def run_inference_on_vertices(vertices_cuda, eigenfunction_idx=0, shape_code=None):
    """
    Run basis set inference on vertices

    Args:
        vertices_cuda: CUDA torch tensor of shape (N, 3) - vertex positions
        eigenfunction_idx: index of eigenfunction to use (default: 0)
        shape_code: optional shape code tensor, if None will use zero code

    Returns:
        CPU numpy array with inference results
    """
    global _basis_set, _device

    if _basis_set is None:
        init_result = initialize_basis_set()
        if "Error" in init_result:
            raise RuntimeError(init_result)

    # Ensure vertices are on the correct device and shape
    if vertices_cuda.ndim == 1:
        vertices_cuda = vertices_cuda.reshape(-1, 3)

    vertices_cuda = vertices_cuda.to(_device)
    num_vertices = vertices_cuda.shape[0]

    # Prepare shape code (use zero code if not provided)
    if shape_code is None:
        # Network expects 5D input: 3D position + 1D shape_code + 1D distance_field
        # So shape_code should be a single scalar value
        shape_code = torch.tensor([0.0], device=_device)
    else:
        # Ensure shape_code is a tensor on the correct device
        if not isinstance(shape_code, torch.Tensor):
            shape_code = torch.tensor([float(shape_code)], device=_device)
        else:
            shape_code = shape_code.to(_device)

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

    # Convert to CPU numpy array and squeeze to 1D
    output_cpu = output.squeeze().cpu().numpy()

    # Log statistics
    print(
        f"Inference completed: {num_vertices} vertices, eigenfunction {eigenfunction_idx}"
    )
    print(f"Output range: [{float(output.min()):.4f}, {float(output.max()):.4f}]")
    print(f"Output mean: {float(output.mean()):.4f}, std: {float(output.std()):.4f}")

    return output_cpu


def get_cuda_tensor():
    """
    Returns a CUDA torch tensor of length 10
    """
    if torch.cuda.is_available():
        tensor = torch.arange(10, dtype=torch.float32, device="cuda")
        print(f"Python: Created CUDA tensor with shape {tensor.shape}")
        return tensor
    else:
        print("Warning: CUDA not available, returning CPU tensor")
        tensor = torch.arange(10, dtype=torch.float32)
        return tensor


def process_vertices_cuda(vertices_cuda):
    """
    Process geometry vertices on GPU
    vertices_cuda: CUDA torch tensor of shape (N*3,) or (N, 3)
    Returns: formatted string with vertex information
    """

    # Reshape if flat array
    if vertices_cuda.ndim == 1:
        vertices_cuda = vertices_cuda.reshape(-1, 3)

    num_vertices = vertices_cuda.shape[0]

    # Perform GPU operations
    # Example: compute bounding box
    min_vals = vertices_cuda.min(dim=0).values
    max_vals = vertices_cuda.max(dim=0).values
    center = vertices_cuda.mean(dim=0)

    # Copy first few vertices to CPU for display
    num_to_print = min(5, num_vertices)
    vertices_cpu = vertices_cuda[:num_to_print].cpu().numpy()

    lines = []
    lines.append(f"Received {num_vertices} vertices on CUDA")
    lines.append(f"Vertices shape: {vertices_cuda.shape}")
    lines.append(f"Device: {vertices_cuda.device}")
    lines.append(
        f"Bounding box: min=({min_vals[0]:.4f}, {min_vals[1]:.4f}, {min_vals[2]:.4f})"
    )
    lines.append(
        f"              max=({max_vals[0]:.4f}, {max_vals[1]:.4f}, {max_vals[2]:.4f})"
    )
    lines.append(f"Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    lines.append(f"First {num_to_print} vertices:")
    for i in range(num_to_print):
        lines.append(
            f"  [{i}] ({vertices_cpu[i][0]:.4f}, {vertices_cpu[i][1]:.4f}, {vertices_cpu[i][2]:.4f})"
        )

    return "\n".join(lines)


def test_save_obj(shape_code_value=0.5, output_filename="test_basis_set.obj"):
    """
    Test the save_obj method of basis_set

    Args:
        shape_code_value: Shape code value(s) to use - can be a single float or list of floats
        output_filename: Output OBJ filename (default: "test_basis_set.obj")

    Returns:
        Status message string
    """
    global _basis_set, _device

    if _basis_set is None:
        # Cannot initialize without model_name from C++
        return "Error: basis_set not initialized. Call initialize_basis_set first."

    try:
        # Create shape code tensor - support both single value and list
        if isinstance(shape_code_value, (list, tuple)):
            shape_code = torch.tensor(shape_code_value, device=_device)
        else:
            shape_code = torch.tensor([shape_code_value], device=_device)

        # Save OBJ file
        output_path = os.path.join(
            r"C:\Users\Pengfei\WorkSpace\Ruzino\Binaries\Debug", output_filename
        )

        print(f"Saving OBJ with shape_code={shape_code_value} to {output_path}")
        
        # Update shape code first
        _basis_set._update_shape_code(shape_code)
        
        # Get geometry from shape_space
        geom = _basis_set.shape_space.geometry
        
        # Get vertices and faces
        vertices = geom.vertices.cpu().numpy() if hasattr(geom.vertices, 'cpu') else geom.vertices
        faces = geom.faces.cpu().numpy() if hasattr(geom.faces, 'cpu') else geom.faces
        
        print(f"Mesh info: {len(vertices)} vertices, {len(faces)} faces")
        
        # Write OBJ file
        with open(output_path, 'w') as f:
            f.write(f"# Generated by Ruzino BssDeducer\n")
            f.write(f"# Shape code: {shape_code_value}\n")
            f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ format is 1-indexed)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        # Verify file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            return f"Successfully saved OBJ file: {output_path} ({file_size} bytes, {len(vertices)} verts, {len(faces)} faces)"
        else:
            return f"Error: OBJ file was not created at {output_path}"

    except Exception as e:
        import traceback

        return f"Error in test_save_obj: {str(e)}\n{traceback.format_exc()}"


def detect_dirichlet_boundary(vertices_cuda, shape_code_value=None, distance_threshold=1e-6):
    """
    Detect which vertices are on Dirichlet boundary using basis_set.is_on_dirichlet_boundary
    
    Args:
        vertices_cuda: CUDA torch tensor of shape (N, 3) - vertex positions
        shape_code_value: Shape code value(s) - can be single float or list of floats
        distance_threshold: Distance threshold for boundary detection (default: 1e-6)
    
    Returns:
        CPU numpy array with 1.0 for boundary vertices, 0.0 for interior vertices
    """
    global _basis_set, _device
    
    if _basis_set is None:
        raise RuntimeError("basis_set not initialized. Call initialize_basis_set first.")
    
    # Ensure vertices are on the correct device and shape
    if vertices_cuda.ndim == 1:
        vertices_cuda = vertices_cuda.reshape(-1, 3)
    
    vertices_cuda = vertices_cuda.to(_device)
    num_vertices = vertices_cuda.shape[0]
    
    # Prepare shape code - same format as inference
    if shape_code_value is None:
        shape_code = torch.tensor([0.0], device=_device)
    elif isinstance(shape_code_value, (list, tuple)):
        shape_code = torch.tensor(shape_code_value, device=_device)
    else:
        shape_code = torch.tensor([float(shape_code_value)], device=_device)
    
    print(f"Detecting Dirichlet boundary for {num_vertices} vertices with shape_code={shape_code_value}, threshold={distance_threshold}")
    
    # Call basis_set's is_on_dirichlet_boundary method directly
    with torch.no_grad():
        boundary_result = _basis_set.is_on_dirichlet_boundary(
            shape_code=shape_code,
            sample_points=vertices_cuda,
            distance_threshold=distance_threshold
        )
    
    # Check if result is boolean or float tensor
    if boundary_result.dtype == torch.bool:
        # Convert boolean to float (1.0 for True, 0.0 for False)
        boundary_values = boundary_result.float().cpu().numpy()
    else:
        # Already float, just ensure it's 0.0 or 1.0
        boundary_values = boundary_result.cpu().numpy()
    
    # Ensure it's 1D
    if boundary_values.ndim > 1:
        boundary_values = boundary_values.squeeze()
    if boundary_values.ndim == 0:
        boundary_values = np.array([float(boundary_values)])
    
    num_boundary = int(np.sum(boundary_values > 0.5))
    print(f"Found {num_boundary} / {num_vertices} vertices on Dirichlet boundary")
    
    return boundary_values
