import os, torch
from lib.data.datasets import InMemoryComplexDataset
from lib.utils.graph_to_complex import convert_graph_dataset_with_paths, convert_graph_dataset_with_rings
from lib.utils.log_utils import makedirs

class SHRIROOKDataset(InMemoryComplexDataset):
    def __init__(self, root, name="SHRIROOK", max_dim=2, num_classes=2,
                 include_down_adj=False, max_ring_size=None, n_jobs=2,
                 init_method="sum", complex_type="path", **kwargs):
        self.name = name
        self._num_classes = num_classes
        self._n_jobs = n_jobs
        self._max_ring_size = max_ring_size
        self.num_node_labels = 1
        self.root = root

        super().__init__(root, max_dim=max_dim, num_classes=num_classes,
                         include_down_adj=include_down_adj,
                         complex_type=complex_type, init_method=init_method)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.name}_complex_list.pt"]

    @property
    def raw_file_names(self):
        return []

    def download(self): pass

    def process(self):
        obj = torch.load(os.path.join(self.root, "raw", "data.pt"))
        graphs = obj["graphs"]

        if self._complex_type == "path":
            complexes, max_dim, _ = convert_graph_dataset_with_paths(
                graphs, max_k=self._max_dim, include_down_adj=self.include_down_adj,
                init_edges=True, init_high_order_paths=True,
                init_method=self._init_method, n_jobs=self._n_jobs
            )
        else:
            raise NotImplementedError(self._complex_type)

        if max_dim != self.max_dim:
            self.max_dim = max_dim
            makedirs(self.processed_dir)

        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])