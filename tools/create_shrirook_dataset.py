import os
from pathlib import Path
import random
import networkx as nx
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def create_rooks_graph(n):
    """
    Create an n x n Rook's graph.

    A Rook's graph represents the possible moves of a rook on an n x n chessboard.
    Each vertex corresponds to a square on the chessboard, and edges connect squares
    that are in the same row or column.

    Parameters:
    n (int): The size of the chessboard (n x n).

    Returns:
    G (NetworkX Graph): The Rook's graph.
    """
    G = nx.Graph()
    
    # Add vertices
    for i in range(n):
        for j in range(n):
            G.add_node((i, j))
    
    # Add edges for rows
    for i in range(n):
        for j1 in range(n):
            for j2 in range(j1 + 1, n): # Avoid duplicate edges
                G.add_edge((i, j1), (i, j2))
    
    # Add edges for columns
    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n): # Avoid duplicate edges
                G.add_edge((i1, j), (i2, j))
    
    return G

def create_shrikhande_graph():
    """
    Create the Shrikhande graph.

    The Shrikhande graph is a strongly regular graph with 16 vertices and 48 edges.
    It can be constructed as a Cayley graph of the group Z4 x Z4 with a specific
    generating set.

    Returns:
    G (NetworkX Graph): The Shrikhande graph.
    """
    G = nx.Graph()
    
    # Define vertices
    vertices = [(i, j) for i in range(4) for j in range(4)]
    G.add_nodes_from(vertices)
    
    # Add edges according to the Shrikhande graph construction
    # Each vertex (i, j) is connected to 6 neighbors
    for i in range(4):
        for j in range(4):
            # Horizontal neighbors (same row, adjacent columns)
            G.add_edge((i, j), (i, (j + 1) % 4))
            G.add_edge((i, j), (i, (j - 1) % 4))
            
            # Vertical neighbors (same column, adjacent rows)
            G.add_edge((i, j), ((i + 1) % 4, j))
            G.add_edge((i, j), ((i - 1) % 4, j))
            
            # Diagonal neighbors (specific pattern for Shrikhande graph)
            G.add_edge((i, j), ((i + 1) % 4, (j + 1) % 4))
            G.add_edge((i, j), ((i + 1) % 4, (j - 1) % 4))
    
    return G

def permute_graph(G, rng=random.Random(42)):
    """
    Return a permuted copy of the input graph G.

    The permutation is defined by a specific mapping of vertices.

    Parameters:
    G (NetworkX Graph): The input graph to be permuted.

    Returns:
    H (NetworkX Graph): The permuted graph.
    """
    nodes = list(G.nodes())
    perm = list(range(len(nodes)))
    rng.shuffle(perm)
    mapping = {nodes[i]: perm[i] for i in range(len(nodes))}

    return nx.relabel_nodes(G, mapping)

def nx_to_pyg(G, y):
    # Ensure nodes are labeled 0..n-1 (important after relabeling)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    n = G.number_of_nodes()
    x = torch.ones(n, 1, dtype=torch.float32)
    y = torch.tensor([y], dtype=torch.long)  # graph label as class index

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=n)

def create_pyg_dataset(n_samples=100, rng=random.Random(42)):
    graphs = []
    rook = create_rooks_graph(4)
    shr = create_shrikhande_graph()

    for _ in range(n_samples):
        graphs.append(nx_to_pyg(permute_graph(rook, rng), y=0))
    for _ in range(n_samples):
        graphs.append(nx_to_pyg(permute_graph(shr, rng), y=1))

    return graphs

if __name__ == "__main__":
    # 1) Generate graphs (PyG Data objects)
    rng = random.Random(42)
    dataset = create_pyg_dataset(n_samples=100, rng=rng)  # total graphs = 2*n_samples

    # 2) Deterministic PyTorch split (indices only)
    n = len(dataset)
    gen = torch.Generator().manual_seed(42)

    train_len = int(0.8 * n)
    val_len   = int(0.1 * n)
    test_len  = n - train_len - val_len

    train_subset, val_subset, test_subset = random_split(
        range(n), [train_len, val_len, test_len], generator=gen
    )

    train_ids = list(train_subset)
    val_ids   = list(val_subset)
    test_ids  = list(test_subset)

    # 3) Save in the dict format your loader expects
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    out_dir = PROJECT_ROOT / "datasets" / "SHRIROOK" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "data.pt"
    obj = {
        "graphs": dataset,
        "splits": {"train": train_ids, "valid": val_ids, "test": test_ids},
    }

    torch.save(obj, out_path)
    print(f"Saved {n} graphs to {out_path}")
    print(f"Split sizes: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")