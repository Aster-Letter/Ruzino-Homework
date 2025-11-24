"""
Ruzino Render Graph API - A high-level Python interface for render node graphs.

This module provides a clean, Falcor-inspired API for constructing and executing
render graphs in Python, similar to how geometry graphs work.

Example:
    from render_graph import RenderGraph
    
    # Create graph
    g = RenderGraph("MyRenderPipeline")
    g.loadConfiguration("render_nodes.json")
    
    # Create nodes
    rng_tex = g.createNode("rng_texture", name="RNG")
    ray_gen = g.createNode("node_render_ray_generation", name="RayGen")
    path_trace = g.createNode("path_tracing", name="PathTrace")
    accumulate = g.createNode("accumulate", name="Accumulate")
    
    # Connect nodes
    g.addEdge(rng_tex, "Random Number", ray_gen, "random seeds")
    g.addEdge(ray_gen, "Rays", path_trace, "Rays")
    g.addEdge(path_trace, "Output", accumulate, "Texture")
    
    # Set parameters
    g.setInput(accumulate, "Max Samples", 16)
    
    # Mark output
    g.markOutput(accumulate, "Accumulated")
    
    # Execute
    g.prepare_and_execute()
    
    # Get result
    result_texture = g.getOutput(accumulate, "Accumulated")
"""

import os
from pathlib import Path
from typing import Any, Optional, Union, Dict, List

# Import required modules
try:
    import nodes_core_py as core
    import nodes_system_py as system
    import hd_USTC_CG_py as renderer
    HAS_RENDERER = True
except ImportError as e:
    HAS_RENDERER = False
    print(f"WARNING: Renderer modules not available: {e}")


class RenderGraph:
    """High-level interface for render node graph construction and execution."""
    
    def __init__(self, name: str = "RenderGraph"):
        """
        Create a new render graph.
        
        Args:
            name: Name of the graph (for debugging/display)
        """
        if not HAS_RENDERER:
            raise RuntimeError(
                "Renderer modules not available. "
                "Make sure hd_USTC_CG_py, nodes_system_py, and nodes_core_py are built."
            )
        
        self.name = name
        self._system: Optional[system.NodeDynamicLoadingSystem] = None
        self._tree: Optional[core.NodeTree] = None
        self._executor: Optional[system.NodeTreeExecutor] = None
        self._initialized = False
        self._node_name_counter = {}
        self._output_marks = []
    
    def _ensure_initialized(self):
        """Ensure the graph system is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "Graph not initialized. Call initialize() or loadConfiguration() first."
            )
    
    def initialize(self, config_path: Optional[str] = None, usd_stage_path: Optional[str] = None) -> 'RenderGraph':
        """
        Initialize the render graph system.
        
        Args:
            config_path: Optional path to render_nodes.json configuration
            usd_stage_path: Optional path to USD stage for scene setup
            
        Returns:
            self for chaining
        """
        if self._system is None:
            self._system = renderer.create_render_system()
        
        if config_path:
            loaded = self._system.load_configuration(config_path)
            if not loaded:
                raise RuntimeError(f"Failed to load configuration from {config_path}")
        
        if not self._initialized:
            self._system.init()
            self._tree = self._system.get_node_tree()
            self._executor = self._system.get_node_tree_executor()
            self._initialized = True
        
        # Set up USD stage if provided
        if usd_stage_path:
            try:
                import stage_py
                stage = stage_py.Stage(str(usd_stage_path))
                payload = stage_py.create_payload_from_stage(stage, "/geom")
                meta_payload = stage_py.create_meta_any_from_payload(payload)
                self._system.set_global_params(meta_payload)
                print(f"✓ USD stage loaded: {usd_stage_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not load USD stage: {e}")
        
        return self
    
    def loadConfiguration(self, config_path: str) -> 'RenderGraph':
        """
        Load node definitions from render_nodes.json.
        
        Args:
            config_path: Path to render_nodes.json
            
        Returns:
            self for chaining
        """
        if not self._initialized:
            return self.initialize(config_path)
        
        loaded = self._system.load_configuration(config_path)
        if not loaded:
            raise RuntimeError(f"Failed to load configuration from {config_path}")
        
        return self
    
    def createNode(self, node_type: str, properties: Optional[dict] = None, 
                   name: Optional[str] = None) -> 'core.Node':
        """
        Create a render node.
        
        Args:
            node_type: Type of node (e.g., "path_tracing", "accumulate")
            properties: Optional properties dictionary
            name: Optional custom name
            
        Returns:
            The created node
        """
        self._ensure_initialized()
        
        if name is None:
            count = self._node_name_counter.get(node_type, 0)
            name = f"{node_type}_{count}"
            self._node_name_counter[node_type] = count + 1
        
        node = self._tree.add_node(node_type)
        if node is None:
            raise RuntimeError(f"Failed to create node of type '{node_type}'")
        
        node.ui_name = name
        
        if properties:
            print(f"Warning: Property setting not yet implemented")
        
        return node
    
    def addEdge(self, 
                from_node: Union['core.Node', str], 
                from_socket: str,
                to_node: Union['core.Node', str],
                to_socket: str) -> 'RenderGraph':
        """
        Connect two nodes.
        
        Args:
            from_node: Source node or name
            from_socket: Output socket name
            to_node: Destination node or name
            to_socket: Input socket name
            
        Returns:
            self for chaining
        """
        self._ensure_initialized()
        
        from_n = self._resolve_node(from_node)
        to_n = self._resolve_node(to_node)
        
        from_sock = from_n.get_output_socket(from_socket)
        to_sock = to_n.get_input_socket(to_socket)
        
        if from_sock is None:
            raise ValueError(f"Socket '{from_socket}' not found on node '{from_n.ui_name}'")
        if to_sock is None:
            raise ValueError(f"Socket '{to_socket}' not found on node '{to_n.ui_name}'")
        
        link = self._tree.add_link(from_sock, to_sock)
        if link is None:
            raise RuntimeError(
                f"Failed to create link from {from_n.ui_name}.{from_socket} "
                f"to {to_n.ui_name}.{to_socket}"
            )
        
        return self
    
    def markOutput(self, node_or_spec: Union['core.Node', str], 
                   socket_name: Optional[str] = None) -> 'RenderGraph':
        """
        Mark an output for tracking.
        
        Args:
            node_or_spec: Node or "node.socket" string
            socket_name: Socket name (if node_or_spec is a node)
            
        Returns:
            self for chaining
        """
        if socket_name is None:
            self._output_marks.append(node_or_spec)
        else:
            if isinstance(node_or_spec, core.Node):
                output_spec = f"{node_or_spec.ui_name}.{socket_name}"
            else:
                output_spec = f"{node_or_spec}.{socket_name}"
            self._output_marks.append(output_spec)
        return self
    
    def setInput(self, 
                 node: Union['core.Node', str], 
                 socket_name: str, 
                 value: Any) -> 'RenderGraph':
        """
        Set an input value on a node.
        
        Args:
            node: Target node or name
            socket_name: Input socket name
            value: Value to set
            
        Returns:
            self for chaining
        """
        self._ensure_initialized()
        
        n = self._resolve_node(node)
        socket = n.get_input_socket(socket_name)
        
        if socket is None:
            raise ValueError(f"Socket '{socket_name}' not found on node '{n.ui_name}'")
        
        meta_value = core.to_meta_any(value)
        self._executor.sync_node_from_external_storage(socket, meta_value)
        return self
    
    def execute(self, required_node: Optional[Union['core.Node', str]] = None) -> 'RenderGraph':
        """
        Execute the render graph.
        
        Args:
            required_node: Optional node to execute up to
            
        Returns:
            self for chaining
        """
        self._ensure_initialized()
        
        req_node = None
        if required_node is not None:
            req_node = self._resolve_node(required_node)
        
        self._executor.execute(self._tree, req_node)
        return self
    
    def prepare_and_execute(self, input_values: Optional[Dict] = None,
                           required_node: Optional[Union['core.Node', str]] = None,
                           auto_require_outputs: bool = True) -> 'RenderGraph':
        """
        Prepare tree, set inputs, and execute.
        
        Args:
            input_values: Dictionary mapping (node, socket_name) to values
            required_node: Optional node to execute up to
            auto_require_outputs: Auto-execute marked outputs
            
        Returns:
            self for chaining
        """
        self._ensure_initialized()
        
        req_node = None
        if required_node is not None:
            req_node = self._resolve_node(required_node)
        elif auto_require_outputs and self._output_marks:
            for mark in reversed(self._output_marks):
                if isinstance(mark, str) and '.' in mark:
                    node_name = mark.split('.')[0]
                    node = self.getNode(node_name)
                    if node:
                        req_node = node
                        break
        
        # Prepare tree
        self._executor.prepare_tree(self._tree, req_node)
        
        # Set inputs
        if input_values:
            for (node, socket_name), value in input_values.items():
                n = self._resolve_node(node)
                socket = n.get_input_socket(socket_name)
                if socket is None:
                    raise ValueError(f"Socket '{socket_name}' not found on node '{n.ui_name}'")
                meta_value = core.to_meta_any(value)
                self._executor.sync_node_from_external_storage(socket, meta_value)
        
        # Execute
        self._executor.execute_tree(self._tree)
        
        return self
    
    def getOutput(self, 
                  node: Union['core.Node', str], 
                  socket_name: str) -> Any:
        """
        Get an output value from a node.
        
        Args:
            node: Source node or name
            socket_name: Output socket name
            
        Returns:
            The output value
        """
        self._ensure_initialized()
        
        n = self._resolve_node(node)
        socket = n.get_output_socket(socket_name)
        
        if socket is None:
            raise ValueError(f"Socket '{socket_name}' not found on node '{n.ui_name}'")
        
        result = core.meta_any()
        self._executor.sync_node_to_external_storage(socket, result)
        
        # Try to extract the value
        type_name = result.type_name()
        if type_name == "int":
            return result.cast_int()
        elif type_name == "float":
            return result.cast_float()
        elif type_name == "bool":
            return result.cast_bool()
        elif "string" in type_name.lower():
            return result.cast_string()
        else:
            # Return raw meta_any for complex types (textures, buffers, etc.)
            return result
    
    def getNode(self, name: str) -> Optional['core.Node']:
        """Get a node by name."""
        self._ensure_initialized()
        
        for node in self._tree.nodes:
            if node.ui_name == name:
                return node
        return None
    
    def _resolve_node(self, node_ref: Union['core.Node', str]) -> 'core.Node':
        """Resolve a node reference."""
        if isinstance(node_ref, str):
            node = self.getNode(node_ref)
            if node is None:
                raise ValueError(f"Node '{node_ref}' not found in graph")
            return node
        return node_ref
    
    def serialize(self) -> str:
        """Serialize the graph to JSON."""
        self._ensure_initialized()
        return self._tree.serialize()
    
    def deserialize(self, json_str: str) -> 'RenderGraph':
        """Deserialize a graph from JSON."""
        self._ensure_initialized()
        self._tree.deserialize(json_str)
        return self
    
    def clear(self) -> 'RenderGraph':
        """Clear all nodes and links."""
        self._ensure_initialized()
        self._tree.clear()
        self._node_name_counter.clear()
        self._output_marks.clear()
        return self
    
    @property
    def nodes(self):
        """Get list of all nodes."""
        self._ensure_initialized()
        return self._tree.nodes
    
    @property
    def links(self):
        """Get list of all links."""
        self._ensure_initialized()
        return self._tree.links
    
    def __repr__(self):
        if not self._initialized:
            return f"RenderGraph('{self.name}', uninitialized)"
        return f"RenderGraph('{self.name}', nodes={len(self.nodes)}, links={len(self.links)})"


# Example usage
if __name__ == "__main__":
    import sys
    
    print("Ruzino Render Graph API")
    print("="*70)
    
    # Find binary directory and add to path
    binary_dir = Path.cwd()
    if not (binary_dir / "render_nodes.json").exists():
        binary_dir = binary_dir / "Binaries" / "Debug"
    
    # Add binary dir to Python path for module imports
    sys.path.insert(0, str(binary_dir))
    
    config_path = binary_dir / "render_nodes.json"
    
    if not config_path.exists():
        print(f"✗ Configuration file not found: {config_path}")
        exit(1)
    
    print(f"✓ Using configuration: {config_path}")
    print()
    
    # Create render graph
    g = RenderGraph("TestPipeline")
    g.loadConfiguration(str(config_path))
    print(f"✓ Created render graph: {g}")
    print()
    
    # Create some basic nodes
    try:
        rng_tex = g.createNode("rng_texture", name="RNGTexture")
        print(f"✓ Created node: {rng_tex.ui_name}")
        
        ray_gen = g.createNode("node_render_ray_generation", name="RayGen")
        print(f"✓ Created node: {ray_gen.ui_name}")
        
        # Connect them
        g.addEdge(rng_tex, "Random Number", ray_gen, "random seeds")
        print(f"✓ Connected: {rng_tex.ui_name} -> {ray_gen.ui_name}")
        
        print()
        print(f"Final graph: {g}")
        print(f"  Nodes: {[n.ui_name for n in g.nodes]}")
        print(f"  Links: {len(g.links)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
