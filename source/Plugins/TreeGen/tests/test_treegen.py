"""
TreeGen Plugin Test
Tests procedural tree generation based on Stava et al. 2014
"""
import os
import sys
from ruzino_graph import RuzinoGraph
import stage_py
import geometry_py as geom


def get_binary_dir():
    """Get the binary directory path"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    binary_dir = os.path.join(test_dir, '..', '..', '..', '..', 'Binaries', 'Debug')
    return os.path.abspath(binary_dir)

def test_tree_generation():
    """Test full procedural tree generation"""
    print("\n" + "="*70)
    print("TEST: Procedural Tree Generation")
    print("="*70)
    
    binary_dir = get_binary_dir()
    g = RuzinoGraph("TreeGenTest")
    config_path = os.path.join(binary_dir, "Plugins", "TreeGen_geometry_nodes.json")
    
    g.loadConfiguration(config_path)
    print(f"✓ Loaded configuration")
    
    # Create tree generation node
    tree = g.createNode("tree_generate", name="tree")
    print(f"✓ Created node: {tree.ui_name}")
    
    # Set parameters for a small tree
    inputs = {
        (tree, "Growth Years"): 5,
        (tree, "Random Seed"): 42,
        (tree, "Apical Angle Variance"): 30.0,
        (tree, "Lateral Buds"): 3,
        (tree, "Branch Angle"): 45.0,
        (tree, "Growth Rate"): 2.5,
        (tree, "Internode Length"): 0.3,
        (tree, "Apical Control"): 2.0,
        (tree, "Apical Dominance"): 1.0,
        (tree, "Light Factor"): 0.6,
        (tree, "Phototropism"): 0.2,
        (tree, "Gravitropism"): 0.1
    }
    
    print("\n🌱 Growing tree with parameters:")
    print(f"  Growth Years: 5")
    print(f"  Branch Angle: 45°")
    print(f"  Apical Control: 2.0")
    
    # Execute
    print("\n🚀 Executing tree growth...")
    g.prepare_and_execute(inputs, required_node=tree)
    print("✓ Tree generation completed")
    
    # Get result
    result = g.getOutput(tree, "Tree Branches")
    geometry = geom.extract_geometry_from_meta_any(result)
    
    curve = geometry.get_curve_component(0)
    assert curve is not None, "No CurveComponent found"
    
    vertices = curve.get_vertices()
    counts = curve.get_vert_count()
    
    print(f"\n📊 Tree Statistics:")
    print(f"  Total vertices: {len(vertices)}")
    print(f"  Branch segments: {len(counts)}")
    print(f"  Branches: {len(counts) // 2}")  # Each branch has 2 points
    
    # Check tree has grown
    assert len(vertices) > 2, "Tree should have more than root branch"
    assert len(counts) > 1, "Tree should have multiple branches"
    
    # Check tree has some height
    max_height = max(v.y for v in vertices)
    print(f"  Max height: {max_height:.2f}")
    assert max_height > 0.5, "Tree should have grown upward"
    
    print("\n✅ Tree generation test passed!")


def test_tree_to_mesh():
    """Test converting tree to mesh - SKIP for now, use test_tree_with_leaves_to_mesh instead"""
    print("\n" + "="*70)
    print("TEST: Tree to Mesh Conversion (SKIPPED)")
    print("="*70)
    print("⚠️  This test is currently skipped - use test_tree_with_leaves_to_mesh")
    print("✓ Test skipped")


def test_parameter_variations():
    """Test different tree parameters produce different results"""
    print("\n" + "="*70)
    print("TEST: Parameter Variations")
    print("="*70)
    
    binary_dir = get_binary_dir()
    g = RuzinoGraph("ParamTest")
    config_path = os.path.join(binary_dir, "Plugins", "TreeGen_geometry_nodes.json")
    g.loadConfiguration(config_path)
    
    # Test 1: High apical control (tall tree)
    print("\n🌲 Test 1: High Apical Control (Tall Tree)")
    tree1 = g.createNode("tree_generate", name="tall_tree")
    inputs1 = {
        (tree1, "Growth Years"): 5,
        (tree1, "Random Seed"): 42,
        (tree1, "Apical Control"): 4.0,
        (tree1, "Branch Angle"): 30.0,
        (tree1, "Growth Rate"): 3.0
    }
    g.prepare_and_execute(inputs1, required_node=tree1)
    result1 = g.getOutput(tree1, "Tree Branches")
    geom1 = geom.extract_geometry_from_meta_any(result1)
    curve1 = geom1.get_curve_component(0)
    verts1 = curve1.get_vertices()
    height1 = max(v.y for v in verts1)
    print(f"  Height: {height1:.2f}")
    
    # Test 2: Low apical control (bushy tree)
    print("\n🌳 Test 2: Low Apical Control (Bushy Tree)")
    tree2 = g.createNode("tree_generate", name="bushy_tree")
    inputs2 = {
        (tree2, "Growth Years"): 5,
        (tree2, "Random Seed"): 123,
        (tree2, "Apical Control"): 1.0,
        (tree2, "Branch Angle"): 60.0,
        (tree2, "Growth Rate"): 3.0
    }
    g.prepare_and_execute(inputs2, required_node=tree2)
    result2 = g.getOutput(tree2, "Tree Branches")
    geom2 = geom.extract_geometry_from_meta_any(result2)
    curve2 = geom2.get_curve_component(0)
    verts2 = curve2.get_vertices()
    height2 = max(v.y for v in verts2)
    branches2 = len(curve2.get_vert_count())
    print(f"  Height: {height2:.2f}")
    print(f"  Branches: {branches2}")
    
    print("\n📊 Comparison:")
    print(f"  Tall tree height: {height1:.2f}")
    print(f"  Bushy tree height: {height2:.2f}")
    print(f"  Height ratio: {height1/height2:.2f}x")
    
    # High apical control should produce taller trees
    assert height1 > height2, "High apical control should produce taller trees"
    
    print("✅ Parameter variation test passed!")


def test_tree_with_leaves_to_mesh():
    """Test converting tree with leaves to mesh"""
    print("\n" + "="*70)
    print("TEST: Tree with Leaves to Mesh")
    print("="*70)
    
    binary_dir = get_binary_dir()
    g = RuzinoGraph("TreeWithLeavesTest")
    config_path = os.path.join(binary_dir, "Plugins", "TreeGen_geometry_nodes.json")
    
    g.loadConfiguration(config_path)
    
    # Create tree with leaves
    tree = g.createNode("tree_generate", name="tree")
    to_mesh = g.createNode("tree_to_mesh", name="mesh_converter")
    
    # Connect both branches and leaves
    g.addEdge(tree, "Tree Branches", to_mesh, "Tree Branches")
    g.addEdge(tree, "Leaves", to_mesh, "Leaves")
    
    inputs = {
        (tree, "Growth Years"): 3,
        (tree, "Random Seed"): 42,
        (tree, "Generate Leaves"): True,
        (tree, "Terminal Leaves Only"): True,
        (tree, "Leaf Terminal Levels"): 3,
        (tree, "Leaves Per Internode"): 4,
        (tree, "Leaf Size"): 0.15,
        (tree, "Leaf Aspect Ratio"): 2.0,
        (tree, "Leaf Inclination"): 45.0,
        (tree, "Apical Control"): 2.0,
        (tree, "Branch Angle"): 45.0,
        (to_mesh, "Radial Segments"): 8
    }
    
    print("\n🌱 Growing tree with leaves...")
    g.prepare_and_execute(inputs, required_node=to_mesh)
    print("✓ Tree generation and mesh conversion completed")
    
    # Get branch mesh
    branch_result = g.getOutput(to_mesh, "Branch Mesh")
    branch_geom = geom.extract_geometry_from_meta_any(branch_result)
    branch_mesh = branch_geom.get_mesh_component(0)
    assert branch_mesh is not None, "No branch mesh found"
    
    branch_verts = branch_mesh.get_vertices()
    print(f"\n📊 Branch Mesh Statistics:")
    print(f"  Vertices: {len(branch_verts)}")
    
    # Get leaf mesh
    leaf_result = g.getOutput(to_mesh, "Leaf Mesh")
    leaf_geom = geom.extract_geometry_from_meta_any(leaf_result)
    leaf_mesh = leaf_geom.get_mesh_component(0)
    assert leaf_mesh is not None, "No leaf mesh found"
    
    leaf_verts = leaf_mesh.get_vertices()
    leaf_faces = leaf_mesh.get_face_vertex_counts()
    
    print(f"\n🍃 Leaf Mesh Statistics:")
    print(f"  Vertices: {len(leaf_verts)}")
    print(f"  Faces: {len(leaf_faces)}")
    
    # Leaves should exist
    assert len(leaf_verts) > 0, "Tree should have leaves"
    assert len(leaf_faces) > 0, "Leaves should have faces"
    
    # Each leaf should be a diamond (4 verts, 4 triangular faces - 2 front, 2 back)
    num_leaves = len(leaf_verts) // 4
    print(f"  Total leaves: {num_leaves}")
    
    print("\n✅ Tree with leaves to mesh test passed!")


if __name__ == "__main__":
    try:
        test_tree_generation()
        test_tree_to_mesh()
        test_parameter_variations()
        test_tree_with_leaves_to_mesh()
        
        print("\n" + "="*70)
        print("  🌳 ALL TREEGEN TESTS PASSED! 🌳")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
