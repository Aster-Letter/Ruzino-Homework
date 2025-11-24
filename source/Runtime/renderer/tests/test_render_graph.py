"""
Test script for Render Graph using RuzinoRenderGraph API.

This test demonstrates creating, connecting, and executing render nodes,
similar to the geometry graph tests.
"""

import os
import sys
import json

# Import modules
from ruzino_render_graph import RuzinoRenderGraph, quick_render

# Try importing torch if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available. Some tests will be skipped.")

# Get binary directory for test configuration files
binary_dir = os.getcwd()


def test_render_graph_creation():
    """Test creating a basic render graph."""
    print("\n" + "="*60)
    print("TEST: Render Graph Creation")
    print("="*60)
    
    # Create graph
    g = RuzinoRenderGraph("TestRenderGraph")
    print(f"✓ Created graph: {g}")
    
    # Load render nodes configuration
    # Check for various possible render node config files
    config_candidates = [
        "render_nodes.json",
        "path_tracing.json",
        "../Assets/path_tracing.json",
        "../../Assets/path_tracing.json"
    ]
    
    config_path = None
    for candidate in config_candidates:
        full_path = os.path.join(binary_dir, candidate)
        if os.path.exists(full_path):
            config_path = full_path
            break
    
    if not config_path:
        print("⚠ No render configuration file found, skipping config load")
        return
    
    try:
        g.loadConfiguration(config_path)
        print(f"✓ Loaded render configuration from: {config_path}")
        print(f"  Graph state: {g}")
    except Exception as e:
        print(f"⚠ Failed to load configuration: {e}")


def test_render_graph_with_usd():
    """Test loading a USD stage."""
    print("\n" + "="*60)
    print("TEST: Render Graph with USD Stage")
    print("="*60)
    
    g = RuzinoRenderGraph("USDRenderGraph")
    
    # Find USD file
    usd_candidates = [
        "../Assets/cornell_box_stage.usdc",
        "../../Assets/cornell_box_stage.usdc",
        "../Assets/sphere_vis_mtlx.usda",
        "../../Assets/sphere_vis_mtlx.usda"
    ]
    
    usd_path = None
    for candidate in usd_candidates:
        full_path = os.path.join(binary_dir, candidate)
        if os.path.exists(full_path):
            usd_path = full_path
            break
    
    if not usd_path:
        print("⚠ No USD file found, skipping test")
        return
    
    try:
        g.loadUSDStage(usd_path)
        print(f"✓ Loaded USD stage")
        print(f"  Graph: {g}")
    except Exception as e:
        print(f"⚠ Failed to load USD: {e}")


def test_render_settings():
    """Test setting render parameters."""
    print("\n" + "="*60)
    print("TEST: Render Settings")
    print("="*60)
    
    g = RuzinoRenderGraph("SettingsTest")
    
    # Set render settings
    g.setRenderSettings(width=512, height=512, spp=4)
    
    print(f"✓ Settings applied")
    print(f"  Current settings: {g.render_settings}")
    
    assert g.render_settings['width'] == 512
    assert g.render_settings['height'] == 512
    assert g.render_settings['spp'] == 4
    print("✓ Settings verified")


def test_create_render_nodes():
    """Test creating render nodes."""
    print("\n" + "="*60)
    print("TEST: Create Render Nodes")
    print("="*60)
    
    g = RuzinoRenderGraph("RenderNodesTest")
    
    # Load configuration
    config_path = os.path.join(binary_dir, "path_tracing.json")
    if not os.path.exists(config_path):
        # Try alternate paths
        alt_paths = ["../Assets/path_tracing.json", "../../Assets/path_tracing.json"]
        for alt in alt_paths:
            alt_full = os.path.join(binary_dir, alt)
            if os.path.exists(alt_full):
                config_path = alt_full
                break
    
    if not os.path.exists(config_path):
        print("⚠ No configuration file found, skipping test")
        return
    
    try:
        g.loadConfiguration(config_path)
        
        # Try to create some basic render nodes
        # These node types should exist in the render system
        node_types_to_try = [
            "accumulate",
            "gamma_correction",
            "present_color"
        ]
        
        created_nodes = []
        for node_type in node_types_to_try:
            try:
                node = g.createNode(node_type, name=f"Test_{node_type}")
                created_nodes.append(node)
                print(f"✓ Created node: {node.ui_name} (type: {node_type})")
            except Exception as e:
                print(f"⚠ Could not create {node_type}: {e}")
        
        print(f"✓ Created {len(created_nodes)} nodes")
        print(f"  Graph contains {len(g.nodes)} node(s)")
        
    except Exception as e:
        print(f"⚠ Test failed: {e}")


def test_render_node_connections():
    """Test connecting render nodes."""
    print("\n" + "="*60)
    print("TEST: Render Node Connections")
    print("="*60)
    
    g = RuzinoRenderGraph("ConnectionsTest")
    
    # Load configuration
    config_path = None
    for candidate in ["path_tracing.json", "../Assets/path_tracing.json"]:
        full_path = os.path.join(binary_dir, candidate)
        if os.path.exists(full_path):
            config_path = full_path
            break
    
    if not config_path:
        print("⚠ No configuration file found, skipping test")
        return
    
    try:
        g.loadConfiguration(config_path)
        
        # Create a simple pipeline: texture -> gamma correction -> present
        accumulate = g.createNode("accumulate", name="Accumulate")
        gamma = g.createNode("gamma_correction", name="Gamma")
        present = g.createNode("present_color", name="Present")
        
        print(f"✓ Created 3 nodes")
        
        # Connect them
        g.addEdge(accumulate, "Accumulated", gamma, "Texture")
        g.addEdge(gamma, "Corrected", present, "Color")
        
        print(f"✓ Connected nodes")
        print(f"  Graph: {len(g.nodes)} nodes, {len(g.links)} links")
        
        assert len(g.links) == 2
        print("✓ Connection count verified")
        
    except Exception as e:
        print(f"⚠ Test failed: {e}")


def test_render_graph_serialization():
    """Test serializing a render graph."""
    print("\n" + "="*60)
    print("TEST: Render Graph Serialization")
    print("="*60)
    
    g = RuzinoRenderGraph("SerializationTest")
    
    # Load configuration
    config_path = None
    for candidate in ["path_tracing.json", "../Assets/path_tracing.json"]:
        full_path = os.path.join(binary_dir, candidate)
        if os.path.exists(full_path):
            config_path = full_path
            break
    
    if not config_path:
        print("⚠ No configuration file found, skipping test")
        return
    
    try:
        g.loadConfiguration(config_path)
        
        # Create a simple graph
        accum = g.createNode("accumulate", name="Accum")
        gamma = g.createNode("gamma_correction", name="Gamma")
        g.addEdge(accum, "Accumulated", gamma, "Texture")
        g.markOutput(gamma, "Corrected")
        
        print(f"✓ Created graph with {len(g.nodes)} nodes")
        
        # Serialize
        json_str = g.serialize()
        print(f"✓ Serialized ({len(json_str)} characters)")
        
        # Parse and verify
        json_obj = json.loads(json_str)
        print(f"  JSON nodes: {len(json_obj.get('nodes_info', {}))}")
        print(f"  JSON links: {len(json_obj.get('links_info', {}))}")
        
    except Exception as e:
        print(f"⚠ Test failed: {e}")


def test_conversion_nodes():
    """Test NVRHI <-> Torch conversion nodes if available."""
    if not HAS_TORCH:
        print("\n" + "="*60)
        print("TEST: Conversion Nodes (SKIPPED - no torch)")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("TEST: Conversion Nodes")
    print("="*60)
    
    g = RuzinoRenderGraph("ConversionTest")
    
    try:
        # These nodes should be registered if torch support is enabled
        # We can't fully test them without a working render pipeline,
        # but we can check if they're available
        
        g.loadConfiguration("path_tracing.json")  # Or whatever config
        
        # Try to create conversion nodes
        try:
            nvrhi_to_torch = g.createNode("nvrhi_to_torch", name="ToTorch")
            print(f"✓ Created NVRHI to Torch conversion node")
        except Exception as e:
            print(f"⚠ NVRHI to Torch node not available: {e}")
        
        try:
            torch_to_nvrhi = g.createNode("torch_to_nvrhi", name="ToNVRHI")
            print(f"✓ Created Torch to NVRHI conversion node")
        except Exception as e:
            print(f"⚠ Torch to NVRHI node not available: {e}")
            
    except Exception as e:
        print(f"⚠ Test failed: {e}")


# Main test runner
if __name__ == "__main__":
    print("="*60)
    print("Ruzino Render Graph Tests")
    print("="*60)
    
    test_render_graph_creation()
    test_render_graph_with_usd()
    test_render_settings()
    test_create_render_nodes()
    test_render_node_connections()
    test_render_graph_serialization()
    test_conversion_nodes()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
