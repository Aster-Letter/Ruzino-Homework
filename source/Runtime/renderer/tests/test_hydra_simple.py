#!/usr/bin/env python3
"""
Simple HydraRenderer test - just test basic initialization and access
"""

import sys
import os
from pathlib import Path

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

os.chdir(str(binary_dir))
sys.path.insert(0, str(binary_dir))

print("Simple HydraRenderer Test")
print("="*50)

try:
    import hd_USTC_CG_py as renderer
    print("✓ Imported hd_USTC_CG_py\n")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 1: Create HydraRenderer
print("Test 1: Create HydraRenderer")
usd_stage = workspace_root / "Assets" / "shader_ball.usdc"

try:
    hydra = renderer.HydraRenderer(str(usd_stage), width=400, height=300)
    print(f"✓ Created HydraRenderer ({hydra.width}x{hydra.height})\n")
except Exception as e:
    print(f"✗ Failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Get NodeSystem
print("Test 2: Get NodeSystem")
try:
    node_system = hydra.get_node_system()
    print(f"✓ Got NodeSystem: {node_system}\n")
except Exception as e:
    print(f"✗ Failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load configuration
print("Test 3: Load configuration")
try:
    config_file = binary_dir / "render_nodes.json"
    node_system.load_configuration(str(config_file))
    print(f"✓ Loaded config from {config_file.name}\n")
except Exception as e:
    print(f"✗ Failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*50)
print("✓ All tests passed!")
print("="*50)
print("\nNext step: Build render graph and test rendering")
