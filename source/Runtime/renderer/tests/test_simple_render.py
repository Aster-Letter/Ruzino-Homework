#!/usr/bin/env python3
"""
Simple Render Graph Test - Test basic node execution without complex rendering

This test creates a minimal graph to verify:
1. Node system initialization
2. Global payload setting
3. Basic node execution (RNG node only)
"""

import sys
import os
from pathlib import Path

# Add binary directory to path
script_dir = Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent.parent.parent
binary_dir = workspace_root / "Binaries" / "Debug"
sys.path.insert(0, str(binary_dir))

print("="*70)
print("Simple Render Graph Test - Basic Execution")
print("="*70)
print(f"Binary directory: {binary_dir}\n")

# Change to binary directory so DLLs can be found
os.chdir(binary_dir)

# Import modules
try:
    import nodes_core_py as core
    import nodes_system_py as system
    import hd_USTC_CG_py as renderer
    print("✓ Successfully imported C++ modules\n")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}\n")
    sys.exit(1)

from render_graph import RenderGraph

# File paths
config_file = binary_dir / "render_nodes.json"

if not config_file.exists():
    print(f"✗ Configuration file not found: {config_file}")
    sys.exit(1)

print(f"✓ Configuration: {config_file.name}\n")

print("="*70)
print("STEP 1: Initialize System")
print("="*70)

try:
    g = RenderGraph("SimpleTest")
    g.loadConfiguration(str(config_file))
    print("✓ Configuration loaded")
    
    # Set global payload
    meta_payload = renderer.create_render_global_payload()
    g.setGlobalParams(payload=meta_payload)
    print("✓ Global payload set\n")
    
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("STEP 2: Create Simple RNG Node")
print("="*70)

try:
    # Create just one RNG node
    rng = g.createNode("rng_texture", name="TestRNG")
    print(f"✓ Created RNG node: {rng.ui_name}")
    
    # Mark its output
    g.markOutput(rng, "Random Number")
    print(f"✓ Marked output: {rng.ui_name}.Random Number\n")
    
except Exception as e:
    print(f"✗ Node creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("STEP 3: Execute Graph")
print("="*70)

try:
    print("Executing...")
    g.prepare_and_execute()
    print("✓ Execution completed!\n")
    
except Exception as e:
    print(f"✗ Execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*70)
print("STEP 4: Check Output")
print("="*70)

try:
    result = g.getOutput(rng, "Random Number")
    print(f"✓ Got output: {type(result)}")
    print(f"  Type: {result.type_name()}")
    print("\n✓ Test passed!")
    
except Exception as e:
    print(f"✗ Getting output failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
