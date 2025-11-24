#!/usr/bin/env python3
"""
Python wrapper for headless rendering using the C++ headless_render executable
This provides a Python interface to the complete USD + Hydra + Node System pipeline
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

def render_scene(
    usd_file: str,
    json_script: str,
    output_image: str,
    width: int = 1920,
    height: int = 1080,
    spp: int = 16,
    headless_exe: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Render a USD scene using the headless renderer
    
    Args:
        usd_file: Path to USD file
        json_script: Path to JSON render node script
        output_image: Output image path (PNG/HDR/EXR)
        width: Image width
        height: Image height  
        spp: Samples per pixel
        headless_exe: Path to headless_render.exe (auto-detected if None)
        
    Returns:
        (success, output_message)
    """
    
    # Find headless_render.exe
    if headless_exe is None:
        script_dir = Path(__file__).parent.resolve()
        workspace_root = script_dir.parent.parent.parent.parent
        headless_exe = workspace_root / "Binaries" / "Debug" / "headless_render.exe"
        
    if not Path(headless_exe).exists():
        return False, f"headless_render.exe not found at: {headless_exe}"
    
    # Build command
    cmd = [
        str(headless_exe),
        str(usd_file),
        str(json_script),
        str(output_image),
        str(width),
        str(height),
        str(spp)
    ]
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            return True, output
        else:
            return False, f"Rendering failed with code {result.returncode}:\n{output}"
            
    except subprocess.TimeoutExpired:
        return False, "Rendering timed out after 5 minutes"
    except Exception as e:
        return False, f"Failed to run headless renderer: {e}"


def main():
    """Example usage"""
    script_dir = Path(__file__).parent.resolve()
    workspace_root = script_dir.parent.parent.parent.parent
    
    usd_file = workspace_root / "Assets" / "shader_ball.usdc"
    json_script = workspace_root / "Binaries" / "Debug" / "render_nodes.json"  
    output_image = workspace_root / "output_test.png"
    
    print("="*70)
    print("Python Headless Render Wrapper")
    print("="*70)
    print(f"USD file: {usd_file}")
    print(f"JSON script: {json_script}")
    print(f"Output: {output_image}\n")
    
    if not usd_file.exists():
        print(f"✗ USD file not found: {usd_file}")
        return 1
        
    if not json_script.exists():
        print(f"✗ JSON script not found: {json_script}")
        return 1
    
    print("Starting render...")
    success, output = render_scene(
        str(usd_file),
        str(json_script),
        str(output_image),
        width=800,
        height=600,
        spp=4
    )
    
    print(output)
    
    if success:
        print(f"\n✓ Render completed successfully!")
        print(f"Output saved to: {output_image}")
        return 0
    else:
        print(f"\n✗ Render failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
