"""
Parameter-level dependency graphs for tool composition.

Based on In-N-Out (Sep 2025): https://arxiv.org/abs/2509.01560

Creates explicit graphs showing which tool outputs can feed into which 
tool inputs based on type compatibility. Makes composition:
- Verifiable: Only valid chains are possible
- Retrievable: Find compatible tools for any output
- Explainable: Understand why tools can/cannot chain

This replaces "LLM imagination" with structured compatibility checking.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

console = Console()
logger = logging.getLogger(__name__)


# Type compatibility rules (output_type -> set of compatible input_types)
TYPE_COMPATIBILITY = {
    # Exact matches
    "string": {"string", "any"},
    "integer": {"integer", "number", "any"},
    "number": {"number", "any"},
    "boolean": {"boolean", "any"},
    "array": {"array", "any"},
    "object": {"object", "any"},
    "null": {"null", "any"},
    "any": {"any", "string", "integer", "number", "boolean", "array", "object"},
    
    # Common semantic types
    "file_handle": {"file_handle", "string", "object", "any"},
    "dataset": {"dataset", "object", "any"},
    "group": {"group", "object", "any"},
    "dataframe": {"dataframe", "object", "any"},
    "path": {"path", "string", "any"},
}


@dataclass
class DependencyEdge:
    """An edge in the dependency graph (output → input connection)."""
    source_tool: str
    source_param: str  # Usually "return" for output
    target_tool: str
    target_param: str
    output_type: str
    input_type: str
    compatibility_score: float = 1.0  # 1.0 = exact match, 0.5 = compatible, 0 = incompatible
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": f"{self.source_tool}.{self.source_param}",
            "target": f"{self.target_tool}.{self.target_param}",
            "output_type": self.output_type,
            "input_type": self.input_type,
            "compatibility": self.compatibility_score,
        }


@dataclass
class ToolNode:
    """A node in the dependency graph representing a tool."""
    tool_id: str
    name: str
    description: str
    inputs: Dict[str, str] = field(default_factory=dict)  # param_name -> type
    output_type: str = "any"
    output_description: str = ""
    category: str = "general"
    
    @property
    def input_types(self) -> Set[str]:
        """Get all input types for this tool."""
        return set(self.inputs.values())
    
    def accepts_type(self, output_type: str) -> List[str]:
        """Return list of parameters that can accept this output type."""
        compatible_params = []
        for param, param_type in self.inputs.items():
            if is_type_compatible(output_type, param_type):
                compatible_params.append(param)
        return compatible_params


class DependencyGraph:
    """
    Parameter-level dependency graph for tool composition.
    
    Builds explicit connections between tool outputs and inputs based on
    type compatibility, enabling:
    - Validation of tool chains before generation
    - Discovery of compatible tool sequences
    - Constraint-based chain generation
    """
    
    def __init__(self, tools: List[Any] = None):
        """
        Initialize dependency graph.
        
        Args:
            tools: List of Tool objects (from tool_schemas.py)
        """
        self.nodes: Dict[str, ToolNode] = {}
        self.edges: List[DependencyEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # tool -> [tools it can feed]
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)  # tool -> [tools that can feed it]
        
        if tools:
            self.build_from_tools(tools)
    
    def build_from_tools(self, tools: List[Any]) -> None:
        """
        Build the dependency graph from Tool objects.
        
        Args:
            tools: List of Tool objects
        """
        console.print(f"[cyan]Building dependency graph for {len(tools)} tools...[/cyan]")
        
        # First pass: create nodes
        for tool in tools:
            inputs = {}
            for param in tool.parameters:
                inputs[param.name] = param.type
            
            # Handle returns - could be string or dict
            if isinstance(tool.returns, dict):
                output_type = tool.returns.get("type", "any")
                output_desc = tool.returns.get("description", "")
            else:
                output_type = "any"  # Legacy string format
                output_desc = str(tool.returns) if tool.returns else ""
            
            node = ToolNode(
                tool_id=tool.tool_id,
                name=tool.name,
                description=tool.description,
                inputs=inputs,
                output_type=output_type,
                output_description=output_desc,
                category=tool.category,
            )
            self.nodes[tool.tool_id] = node
        
        # Second pass: create edges
        for source_id, source in self.nodes.items():
            for target_id, target in self.nodes.items():
                if source_id == target_id:
                    continue  # Skip self-loops
                
                # Check if source output can feed any target input
                compatible_params = target.accepts_type(source.output_type)
                
                for param in compatible_params:
                    score = compute_compatibility_score(
                        source.output_type, 
                        target.inputs[param]
                    )
                    
                    edge = DependencyEdge(
                        source_tool=source_id,
                        source_param="return",
                        target_tool=target_id,
                        target_param=param,
                        output_type=source.output_type,
                        input_type=target.inputs[param],
                        compatibility_score=score,
                    )
                    self.edges.append(edge)
                    
                    # Update adjacency lists
                    if target_id not in self.adjacency[source_id]:
                        self.adjacency[source_id].append(target_id)
                    if source_id not in self.reverse_adjacency[target_id]:
                        self.reverse_adjacency[target_id].append(source_id)
        
        console.print(f"[green]✓ Built graph: {len(self.nodes)} nodes, {len(self.edges)} edges[/green]")
    
    def get_successors(self, tool_id: str) -> List[str]:
        """Get all tools that can receive output from this tool."""
        return self.adjacency.get(tool_id, [])
    
    def get_predecessors(self, tool_id: str) -> List[str]:
        """Get all tools whose output can feed this tool."""
        return self.reverse_adjacency.get(tool_id, [])
    
    def get_compatible_chains(
        self,
        start_tool: str,
        max_length: int = 4,
        min_length: int = 2,
    ) -> List[List[str]]:
        """
        Find all valid tool chains starting from a given tool.
        
        Args:
            start_tool: Starting tool ID
            max_length: Maximum chain length
            min_length: Minimum chain length to include
            
        Returns:
            List of valid tool chains (each is a list of tool IDs)
        """
        if start_tool not in self.nodes:
            return []
        
        chains = []
        self._dfs_chains(start_tool, [start_tool], max_length, min_length, chains)
        return chains
    
    def _dfs_chains(
        self,
        current: str,
        path: List[str],
        max_length: int,
        min_length: int,
        chains: List[List[str]],
    ) -> None:
        """DFS to find all valid chains."""
        if len(path) >= min_length:
            chains.append(path.copy())
        
        if len(path) >= max_length:
            return
        
        for successor in self.get_successors(current):
            if successor not in path:  # Avoid cycles
                path.append(successor)
                self._dfs_chains(successor, path, max_length, min_length, chains)
                path.pop()
    
    def validate_chain(self, tool_ids: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate that a tool chain is compatible.
        
        Args:
            tool_ids: List of tool IDs in order
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        for i, tool_id in enumerate(tool_ids):
            if tool_id not in self.nodes:
                issues.append(f"Unknown tool: {tool_id}")
                continue
            
            if i > 0:
                prev_tool = tool_ids[i - 1]
                if prev_tool in self.nodes and tool_id not in self.adjacency.get(prev_tool, []):
                    issues.append(
                        f"Incompatible: {prev_tool} output cannot feed {tool_id}"
                    )
        
        return len(issues) == 0, issues
    
    def find_bridges(
        self,
        tool_a: str,
        tool_b: str,
        max_hops: int = 2,
    ) -> List[List[str]]:
        """
        Find intermediate tools that can connect two incompatible tools.
        
        Args:
            tool_a: Source tool ID
            tool_b: Target tool ID
            max_hops: Maximum number of intermediate tools
            
        Returns:
            List of bridge paths (each is list of intermediate tool IDs)
        """
        if tool_a not in self.nodes or tool_b not in self.nodes:
            return []
        
        # Direct connection?
        if tool_b in self.adjacency.get(tool_a, []):
            return [[]]  # Empty bridge - direct connection possible
        
        bridges = []
        
        # One hop bridges
        for mid in self.get_successors(tool_a):
            if tool_b in self.adjacency.get(mid, []):
                bridges.append([mid])
        
        # Two hop bridges (if needed and allowed)
        if max_hops >= 2 and not bridges:
            for mid1 in self.get_successors(tool_a):
                for mid2 in self.adjacency.get(mid1, []):
                    if mid2 != tool_a and tool_b in self.adjacency.get(mid2, []):
                        bridges.append([mid1, mid2])
        
        return bridges
    
    def get_entry_points(self) -> List[str]:
        """Get tools with no required inputs (can start a chain)."""
        entry_points = []
        for tool_id, node in self.nodes.items():
            # Tool is an entry point if it has no required inputs
            # or only has inputs with default values
            # For simplicity, we check if it has few inputs
            if len(node.inputs) <= 1:
                entry_points.append(tool_id)
        return entry_points
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.nodes:
            return {"nodes": 0, "edges": 0}
        
        in_degrees = [len(self.reverse_adjacency.get(n, [])) for n in self.nodes]
        out_degrees = [len(self.adjacency.get(n, [])) for n in self.nodes]
        
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "avg_in_degree": sum(in_degrees) / len(in_degrees),
            "avg_out_degree": sum(out_degrees) / len(out_degrees),
            "max_in_degree": max(in_degrees),
            "max_out_degree": max(out_degrees),
            "entry_points": len(self.get_entry_points()),
            "categories": len(set(n.category for n in self.nodes.values())),
        }
    
    def print_summary(self) -> None:
        """Print a summary of the dependency graph."""
        stats = self.get_statistics()
        
        table = Table(title="Dependency Graph Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Tools (nodes)", str(stats["nodes"]))
        table.add_row("Dependencies (edges)", str(stats["edges"]))
        table.add_row("Avg connections out", f"{stats['avg_out_degree']:.1f}")
        table.add_row("Avg connections in", f"{stats['avg_in_degree']:.1f}")
        table.add_row("Entry points", str(stats["entry_points"]))
        table.add_row("Categories", str(stats["categories"]))
        
        console.print(table)
    
    def print_tool_connections(self, tool_id: str) -> None:
        """Print connections for a specific tool."""
        if tool_id not in self.nodes:
            console.print(f"[red]Tool not found: {tool_id}[/red]")
            return
        
        node = self.nodes[tool_id]
        
        tree = Tree(f"[bold]{node.name}[/bold] ({node.output_type})")
        
        # Predecessors
        preds = self.get_predecessors(tool_id)
        if preds:
            pred_branch = tree.add("[cyan]Can receive from:[/cyan]")
            for p in preds:
                pred_node = self.nodes[p]
                pred_branch.add(f"{pred_node.name} ({pred_node.output_type})")
        
        # Successors
        succs = self.get_successors(tool_id)
        if succs:
            succ_branch = tree.add("[green]Can feed into:[/green]")
            for s in succs:
                succ_node = self.nodes[s]
                compatible_params = succ_node.accepts_type(node.output_type)
                succ_branch.add(f"{succ_node.name} → {', '.join(compatible_params)}")
        
        console.print(tree)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format."""
        return {
            "nodes": {
                tool_id: {
                    "name": node.name,
                    "description": node.description,
                    "inputs": node.inputs,
                    "output_type": node.output_type,
                    "category": node.category,
                }
                for tool_id, node in self.nodes.items()
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "adjacency": dict(self.adjacency),
            "statistics": self.get_statistics(),
        }
    
    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        console.print(f"[green]✓ Saved dependency graph to {path}[/green]")


def is_type_compatible(output_type: str, input_type: str) -> bool:
    """
    Check if an output type is compatible with an input type.
    
    Args:
        output_type: Type of the output
        input_type: Type expected by the input
        
    Returns:
        True if compatible, False otherwise
    """
    # Normalize types
    output_type = output_type.lower()
    input_type = input_type.lower()
    
    # Exact match
    if output_type == input_type:
        return True
    
    # Check compatibility rules
    compatible_inputs = TYPE_COMPATIBILITY.get(output_type, set())
    return input_type in compatible_inputs


def compute_compatibility_score(output_type: str, input_type: str) -> float:
    """
    Compute a compatibility score between output and input types.
    
    Args:
        output_type: Type of the output
        input_type: Type expected by the input
        
    Returns:
        Score from 0.0 (incompatible) to 1.0 (exact match)
    """
    output_type = output_type.lower()
    input_type = input_type.lower()
    
    if output_type == input_type:
        return 1.0
    
    if input_type == "any":
        return 0.8  # Accepts anything, but not ideal
    
    compatible_inputs = TYPE_COMPATIBILITY.get(output_type, set())
    if input_type in compatible_inputs:
        return 0.6  # Compatible but not exact
    
    return 0.0  # Incompatible


def build_graph_from_tools(tools: List[Any]) -> DependencyGraph:
    """
    Convenience function to build a dependency graph from tools.
    
    Args:
        tools: List of Tool objects
        
    Returns:
        Populated DependencyGraph
    """
    return DependencyGraph(tools)
