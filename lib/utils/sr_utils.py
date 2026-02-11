import glob
import os

import networkx as nx
import torch
from torch_geometric.utils import to_undirected


def sr_families():
    return [
        'sr16622', 'sr251256', 'sr261034',
        'sr281264', 'sr291467', 'sr351668',
        'sr351899', 'sr361446', 'sr401224'
    ]


def rnd_families(root_dir=None):
    """Get list of random graph families from RANDOM-GRAPHS/raw directory."""
    if root_dir is None:
        # Try to find the datasets directory
        from definitions import ROOT_DIR
        root_dir = os.path.join(ROOT_DIR, 'datasets', 'RANDOM-GRAPHS', 'raw')
    
    if not os.path.exists(root_dir):
        # Return empty list if directory doesn't exist yet
        return []
    
    # Find all .g6 files and extract family names
    g6_files = glob.glob(os.path.join(root_dir, 'rnd*.g6'))
    families = []
    for filepath in sorted(g6_files):
        basename = os.path.basename(filepath)
        family_name = basename.replace('.g6', '')
        families.append(family_name)
    
    return families


def load_sr_dataset(path):
    """Load the Strongly Regular Graph Dataset from the supplied path."""
    nx_graphs = nx.read_graph6(path)
    graphs = list()
    for nx_graph in nx_graphs:
        n = nx_graph.number_of_nodes()
        edge_index = to_undirected(torch.tensor(list(nx_graph.edges()), dtype=torch.long).transpose(1,0))
        graphs.append((edge_index, n))
        
    return graphs