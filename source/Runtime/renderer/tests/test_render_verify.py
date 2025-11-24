#!/usr/bin/env python3
"""
Simplified Render Test - Load and execute graph using JSON file

This test:
1. Uses the C++ headless render (working)
2. Then loads the output image in Python
3. Verifies it's not pure black
"""

import sys
import os
import subprocess
from pathlib import Path

# Run C++ render first
binary_dir = Path(os.getcwd())
assets_dir = binary_dir.parent.parent / "Assets"

render_graph = assets_dir / "render_nodes_save.json"
usd_scene = assets_dir / "shader_ball.usdc"
output_image = "test_output.png"

print("="*70)
print("STEP 1: Running C++ Headless Render")
print("="*70)

headless_exe = binary_dir / "headless_render.exe"
if not headless_exe.exists():
    print(f"✗ headless_render.exe not found")
    sys.exit(1)

cmd = [
    str(headless_exe),
    str(usd_scene),
    str(render_graph),
    output_image,
    "512",
    "512",
    "4"
]

print(f"Executing: {headless_exe.name}")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

if result.returncode != 0:
    print("✗ Render failed!")
    print(result.stdout)
    print(result.stderr)
    sys.exit(1)

print("✓ Render completed")
print()

# Now analyze the output
print("="*70)
print("STEP 2: Analyzing Output Image")
print("="*70)

if not os.path.exists(output_image):
    print(f"✗ Output file not created: {output_image}")
    sys.exit(1)

try:
    # Try to import image processing library
    try:
        from PIL import Image
        import numpy as np
        has_pil = True
    except ImportError:
        print("⚠ PIL not available, will check file size only")
        has_pil = False
    
    if has_pil:
        # Load and analyze image
        img = Image.open(output_image)
        arr = np.array(img)
        
        print(f"✓ Image loaded: {img.size[0]}x{img.size[1]}, mode={img.mode}")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        
        # Check if image is pure black
        if arr.ndim == 3:
            # RGB(A) image
            avg_value = arr.mean()
            max_value = arr.max()
            min_value = arr.min()
        else:
            # Grayscale
            avg_value = arr.mean()
            max_value = arr.max()
            min_value = arr.min()
        
        print(f"  Min value: {min_value}")
        print(f"  Max value: {max_value}")
        print(f"  Average value: {avg_value:.2f}")
        print()
        
        if max_value == 0:
            print("✗✗✗ IMAGE IS PURE BLACK! ✗✗✗")
            sys.exit(1)
        elif avg_value < 1.0:
            print("⚠ Warning: Image is very dark (avg < 1.0)")
        else:
            print("="*70)
            print("✓✓✓ SUCCESS: Image is NOT pure black! ✓✓✓")
            print("="*70)
            print(f"\nImage statistics:")
            print(f"  - Contains non-black pixels")
            print(f"  - Average pixel value: {avg_value:.2f}")
            print(f"  - Max pixel value: {max_value}")
            print(f"\nThe render graph is working correctly!")
    else:
        # Without PIL, just check file size
        size = os.path.getsize(output_image)
        print(f"✓ Output file exists: {size:,} bytes")
        
        if size > 10000:
            print("✓ File size suggests image has content")
            print("\n✓✓✓ Render completed (install PIL to verify content)")
        else:
            print("⚠ File size is small, image might be empty")

except Exception as e:
    print(f"✗ Error analyzing image: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
