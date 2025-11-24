#!/usr/bin/env python3
"""
Test HydraRenderer - Python render graph with USD/Hydra integration

This demonstrates:
1. Load USD stage through Hydra
2. Get NodeSystem from render delegate  
3. Build graph in Python
4. Execute through Hydra
5. Get output texture in Python
"""

import sys
import os
from pathlib import Path
import numpy as np

# Setup paths
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent.parent.parent
binary_dir = workspace_root / "Binaries" / "Debug"

# Set environment variables for USD and MaterialX (CRITICAL!)
os.environ['PXR_USD_WINDOWS_DLL_PATH'] = str(binary_dir)
print(f"Set PXR_USD_WINDOWS_DLL_PATH={binary_dir}")

# Set MaterialX standard library path for USD MaterialX plugin
mtlx_stdlib = binary_dir / "libraries"
if mtlx_stdlib.exists():
    os.environ['PXR_MTLX_STDLIB_SEARCH_PATHS'] = str(mtlx_stdlib)
    print(f"Set PXR_MTLX_STDLIB_SEARCH_PATHS={mtlx_stdlib}")
else:
    print(f"Warning: MaterialX stdlib not found at {mtlx_stdlib}")

# MUST run from binary directory where render_nodes.json is located
os.chdir(str(binary_dir))
sys.path.insert(0, str(binary_dir))

print("="*70)
print("HydraRenderer Test - Python Graph + USD/Hydra")
print("="*70)
print(f"Working directory: {os.getcwd()}")
print(f"Binary directory: {binary_dir}")

# Verify render_nodes.json exists
config_path = Path(os.getcwd()) / "render_nodes.json"
if not config_path.exists():
    print(f"✗ ERROR: render_nodes.json not found in {os.getcwd()}")
    sys.exit(1)
print(f"✓ Found render_nodes.json\n")

# Import modules
try:
    import hd_USTC_CG_py as renderer
    print("✓ Successfully imported hd_USTC_CG_py\n")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# File paths
usd_stage = workspace_root / "Assets" / "shader_ball.usdc"

if not usd_stage.exists():
    print(f"✗ USD stage not found: {usd_stage}")
    sys.exit(1)

print(f"✓ USD stage: {usd_stage.name}\n")

print("="*70)
print("STEP 1: Initialize HydraRenderer")
print("="*70)

try:
    # Create Hydra renderer (loads USD, initializes Hydra + our render delegate)
    hydra = renderer.HydraRenderer(str(usd_stage), width=800, height=600)
    print(f"✓ HydraRenderer created ({hydra.width}x{hydra.height})")
    
    # Get the node system from the render delegate
    node_system = hydra.get_node_system()
    print(f"✓ Got NodeSystem from render delegate")
    
    # Load configuration manually in Python
    config_file = binary_dir / "render_nodes.json"
    node_system.load_configuration(str(config_file))
    print(f"✓ Loaded configuration: {config_file.name}\n")
    
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("STEP 2: Build Render Graph in Python")
print("="*70)

try:
    # Import the render graph helper module
    sys.path.insert(0, str(binary_dir))
    import render_graph
    
    # Use the render graph API to build the pipeline
    graph = render_graph.RenderGraph(node_system)
    
    # Load configuration
    config_file = binary_dir / "render_nodes.json"
    if not graph.loadConfiguration(str(config_file)):
        raise RuntimeError(f"Failed to load configuration from {config_file}")
    print(f"  ✓ Loaded configuration from {config_file.name}")
    
    # Create nodes using the API
    print("Creating nodes...")
    rng = graph.createNode("rng_texture", "RNG")
    ray_gen = graph.createNode("node_render_ray_generation", "RayGen")
    path_trace = graph.createNode("path_tracing", "PathTracer")
    accumulate = graph.createNode("accumulate", "Accumulate")
    rng_buffer = graph.createNode("rng_buffer", "RNGBuffer")
    
    print("  ✓ Created all nodes")
    
    # Connect nodes
    print("\nConnecting nodes...")
    graph.addEdge(rng, "Random Number", ray_gen, "random seeds")
    graph.addEdge(ray_gen, "Pixel Target", path_trace, "Pixel Target")
    graph.addEdge(ray_gen, "Rays", path_trace, "Rays")
    graph.addEdge(rng_buffer, "Random Number", path_trace, "Random Seeds")
    graph.addEdge(path_trace, "Output", accumulate, "Texture")
    print("  ✓ Connected all nodes")
    
    # Mark output
    graph.markOutput(accumulate, "Accumulated")
    print(f"\n✓ Graph built successfully\n")
    
except Exception as e:
    print(f"\n✗ Failed to build graph: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("STEP 3: Render through Hydra")
print("="*70)

try:
    # Render multiple samples
    num_samples = 4
    print(f"Rendering {num_samples} samples...")
    
    for i in range(num_samples):
        print(f"  Sample {i+1}/{num_samples}...")
        hydra.render()
    
    print("✓ Rendering complete\n")
    
except Exception as e:
    print(f"✗ Rendering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("STEP 4: Get Output Texture")
print("="*70)

try:
    # Get the rendered texture
    print("Reading output texture...")
    texture_data = hydra.get_output_texture()
    print(f"✓ Got texture data: {len(texture_data)} floats")
    
    # Convert to numpy array
    img_array = np.array(texture_data).reshape(hydra.height, hydra.width, 4)
    print(f"✓ Reshaped to: {img_array.shape}")
    
    # Check if it's not all black
    mean_value = img_array[:, :, :3].mean()
    print(f"✓ Mean pixel value: {mean_value:.4f}")
    
    if mean_value > 0.001:
        print("✓ Image has content (not all black)")
    else:
        print("⚠ Warning: Image appears to be all black")
    
    # Save as PNG
    try:
        from PIL import Image
        
        # Flip vertically and convert to uint8
        img_uint8 = (np.clip(img_array[:, :, :3], 0, 1) * 255).astype(np.uint8)
        img_uint8 = np.flipud(img_uint8)
        
        output_path = workspace_root / "output_hydra_test.png"
        Image.fromarray(img_uint8).save(output_path)
        print(f"✓ Saved image to: {output_path}")
    except ImportError:
        print("⚠ PIL not available, skipping image save")
    
except Exception as e:
    print(f"✗ Failed to get output: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ Test completed successfully!")
print("="*70)
