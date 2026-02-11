#!/usr/bin/env python3
"""
Generate random graph families for isomorphism testing.

This script creates random graphs with specified node counts and varying edge densities.
The edge density is sampled from a truncated normal distribution centered at 0.5.
Multiple graphs are generated for each node count to create an isomorphism testing dataset
similar to SR-GRAPHS.

Usage:
    python datasets/random_graphs.py --nodes 10,15,20 --num_graphs 10 --seed 42
    python datasets/random_graphs.py --nodes 16,25,28 --num_graphs 20 --seed None
"""

import argparse
import os
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.stats import truncnorm
import datetime

def get_truncated_normal(mean=0.5, sd=0.15, low=0.1, upp=0.9):
    """
    Create a truncated normal distribution for edge density sampling.
    
    Args:
        mean: Center of distribution (default 0.5)
        sd: Standard deviation (default 0.15)
        low: Lower bound (default 0.1)
        upp: Upper bound (default 0.9)
    
    Returns:
        A truncated normal distribution object
    """
    a = (low - mean) / sd
    b = (upp - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd)


def generate_random_graph(num_nodes, edge_density, seed=None):
    """
    Generate a random graph with specified node count and edge density.
    
    Args:
        num_nodes: Number of nodes in the graph
        edge_density: Fraction of possible edges to include (0 to 1)
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX graph object
    """
    rng = np.random.RandomState(seed)
    
    # Create empty graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Generate all possible edges
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
    
    # Determine number of edges to add
    num_edges = int(len(possible_edges) * edge_density)
    
    # Randomly select edges
    selected_edges = rng.choice(len(possible_edges), size=num_edges, replace=False)
    edges_to_add = [possible_edges[i] for i in selected_edges]
    
    G.add_edges_from(edges_to_add)
    
    return G


def generate_graph_family(num_nodes, num_graphs, base_seed=None, 
                         mean_density=0.5, density_std=0.15):
    """
    Generate a family of random graphs with the same node count and same number of edges.
    All graphs in the family have identical node count and edge count, but different edge arrangements.
    
    Args:
        num_nodes: Number of nodes in each graph
        num_graphs: Number of graphs to generate in this family
        base_seed: Base seed for reproducibility (None for random)
        mean_density: Mean edge density (default 0.5)
        density_std: Standard deviation of edge density (default 0.15)
    
    Returns:
        List of NetworkX graph objects
    """
    if base_seed is not None:
        np.random.seed(base_seed)
    
    # Create truncated normal distribution for edge densities
    tn = get_truncated_normal(mean=mean_density, sd=density_std, low=0.1, upp=0.9)
    
    # Sample ONE edge density for the entire family (all graphs will have same number of edges)
    edge_density = tn.rvs()
    
    # Calculate number of edges for this family
    max_edges = num_nodes * (num_nodes - 1) / 2
    num_edges_family = int(max_edges * edge_density)
    
    print(f"  Family parameters: {num_nodes} nodes, {num_edges_family} edges "
          f"(density: {edge_density:.3f})")
    
    graphs = []
    for i in range(num_graphs):
        # Generate seed for this specific graph
        graph_seed = None if base_seed is None else base_seed + i * 1000
        
        # Generate graph with fixed edge density
        G = generate_random_graph(num_nodes, edge_density, seed=graph_seed)
        
        graphs.append(G)
        
        # Print progress
        print(f"  Graph {i+1}/{num_graphs}: Generated")
    
    return graphs


def save_family_to_g6(graphs, output_path):
    """
    Save a family of graphs to graph6 format file.
    
    Args:
        graphs: List of NetworkX graph objects
        output_path: Path to output .g6 file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write graphs in graph6 format (one per line)
    with open(output_path, 'wb') as f:
        for graph in graphs:
            # Write each graph in graph6 format followed by newline
            graph6_string = nx.to_graph6_bytes(graph, header=False)
            f.write(graph6_string)
    
    print(f"Saved {len(graphs)} graphs to {output_path}")


def create_random_graph_dataset(node_counts, num_graphs_per_family=10, seed=None,
                                mean_density=0.5, density_std=0.15, output_dir=None):
    """
    Create a complete random graph dataset with multiple families.
    
    Args:
        node_counts: List of node counts for different families
        num_graphs_per_family: Number of graphs to generate per family
        seed: Base seed for reproducibility (None for random)
        mean_density: Mean edge density
        density_std: Standard deviation of edge density
        output_dir: Output directory (default: datasets/RANDOM-GRAPHS/raw)
    
    Returns:
        Dictionary mapping family names to lists of graphs
    """
    if output_dir is None:
        # Default to datasets/RANDOM-GRAPHS/raw structure
        script_dir = Path(__file__).parent
        output_dir = script_dir / "RANDOM-GRAPHS" / "raw"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating Random Graph Dataset")
    print(f"=" * 60)
    print(f"Node counts: {node_counts}")
    print(f"Graphs per family: {num_graphs_per_family}")
    print(f"Seed: {seed}")
    print(f"Mean edge density: {mean_density}")
    print(f"Density std: {density_std}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)
    
    all_families = {}
    
    for node_count in node_counts:
        print(f"\nGenerating family with {node_count} nodes...")
        
        # Generate seed for this family
        family_seed = None if seed is None else seed + node_count * 10000
        
        # Generate graphs
        graphs = generate_graph_family(
            num_nodes=node_count,
            num_graphs=num_graphs_per_family,
            base_seed=family_seed,
            mean_density=mean_density,
            density_std=density_std
        )
        
        # Create family name (similar to SR naming: rnd<num_nodes>)
        family_name = f"rnd{node_count:02d}"
        
        # Save to file
        output_path = output_dir / f"{family_name}.g6"
        save_family_to_g6(graphs, output_path)
        
        all_families[family_name] = graphs
    
    # Create README
    readme_path = output_dir.parent / "README.md"
    create_readme(readme_path, node_counts, num_graphs_per_family, seed, 
                  mean_density, density_std)
    
    print(f"\n" + "=" * 60)
    print(f"Dataset creation complete!")
    print(f"Generated {len(all_families)} families with {num_graphs_per_family} graphs each")
    print(f"Total graphs: {len(all_families) * num_graphs_per_family}")
    print(f"=" * 60)
    
    return all_families


def create_readme(readme_path, node_counts, num_graphs_per_family, seed,
                 mean_density, density_std):
    """Create a README file describing the dataset."""
    content = f"""# Random Graph Dataset

This folder contains families of random graphs for isomorphism testing.

## Generation Parameters

- **Node counts**: {', '.join(map(str, node_counts))}
- **Graphs per family**: {num_graphs_per_family}
- **Seed**: {seed if seed is not None else 'Random (not reproducible)'}
- **Edge density distribution**: Truncated Normal(μ={mean_density}, σ={density_std}, range=[0.1, 0.9])

## Dataset Structure

Each family is stored in `./raw/rnd<num_nodes>.g6` format, where `<num_nodes>` is the number of nodes (zero-padded to 2 digits).

Graphs within each family have the same number of nodes and the same number of edges (edge density sampled once per family from a truncated normal distribution), but different edge arrangements. This creates non-isomorphic graphs with identical basic properties for testing graph isomorphism and GNN expressivity.

## Families

{chr(10).join([f"- **rnd{n:02d}**: {num_graphs_per_family} graphs with {n} nodes" for n in node_counts])}

## Usage

This dataset can be used with the Path Complex Networks codebase for isomorphism testing tasks, similar to the SR-GRAPHS dataset.

## Generation Script

Generated using `datasets/random_graphs.py` on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"\nCreated README at {readme_path}")


def parse_node_list(node_str):
    """Parse comma-separated node counts."""
    try:
        nodes = [int(n.strip()) for n in node_str.split(',')]
        # Validate node counts
        for n in nodes:
            if n < 3:
                raise ValueError(f"Node count must be at least 3, got {n}")
        return nodes
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid node count list: {e}")


def parse_seed(seed_str):
    """Parse seed argument (int or None)."""
    if seed_str.lower() in ['none', 'null', 'random']:
        return None
    try:
        return int(seed_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Seed must be an integer or 'None', got {seed_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random graph families for isomorphism testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 3 families with 10 graphs each
  python datasets/random_graphs.py --nodes 10,15,20 --num_graphs 10 --seed 42
  
  # Generate with random seed (not reproducible)
  python datasets/random_graphs.py --nodes 16,25,28 --num_graphs 20 --seed None
  
  # Generate with custom density parameters
  python datasets/random_graphs.py --nodes 12,16,20 --num_graphs 15 --mean_density 0.6 --density_std 0.2
        """
    )
    
    parser.add_argument('--nodes', type=parse_node_list, required=True,
                       help='Comma-separated list of node counts for graph families (e.g., "10,15,20")')
    parser.add_argument('--num_graphs', type=int, default=10,
                       help='Number of graphs to generate per family (default: 10)')
    parser.add_argument('--seed', type=parse_seed, default=42,
                       help='Random seed for reproducibility, or "None" for random (default: 42)')
    parser.add_argument('--mean_density', type=float, default=0.5,
                       help='Mean edge density for truncated normal distribution (default: 0.5)')
    parser.add_argument('--density_std', type=float, default=0.15,
                       help='Standard deviation of edge density distribution (default: 0.15)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: datasets/RANDOM-GRAPHS/raw)')
    
    args = parser.parse_args()
    
    # Create the dataset
    create_random_graph_dataset(
        node_counts=args.nodes,
        num_graphs_per_family=args.num_graphs,
        seed=args.seed,
        mean_density=args.mean_density,
        density_std=args.density_std,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
