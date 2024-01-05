import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import copy
import glob
import os.path as osp
from typing import Callable, Optional

import exp_utils
import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader
from torch_geometric.io import read_txt_array
from egnn.transform.preprocess import *

import os

import torch
import scipy.io

from torch_geometric.data import (Data, InMemoryDataset)
from typing import Callable, List, Optional

class TOSCA_PT(InMemoryDataset):
    """The TOSCA dataset from the `"Numerical Geometry of Non-Ridig Shapes"
    <https://www.amazon.com/Numerical-Geometry-Non-Rigid-Monographs-Computer/
    dp/0387733000>`_ book, containing 80 meshes.
    Meshes within the same category have the same triangulation and an equal
    number of vertices numbered in a compatible way.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (str): Root directory where the dataset should be saved.
        categories (list, optional): List of categories to include in the
            dataset. Can include the categories :obj:`"Cat"`, :obj:`"Centaur"`,
            :obj:`"David"`, :obj:`"Dog"`, :obj:`"Gorilla"`, :obj:`"Horse"`,
            :obj:`"Michael"`, :obj:`"Victoria"`, :obj:`"Wolf"`. If set to
            :obj:`None`, the dataset will contain all categories. (default:
            :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'http://tosca.cs.technion.ac.il/data/toscahires-asci.zip'

    categories = [
        'cat', 'centaur', 'david', 'dog', 'gorilla', 'horse', 'michael',
        'victoria', 'wolf'
    ]

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        categories = self.categories if categories is None else categories
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = categories
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['cat0.vert', 'cat0.tri']

    @property
    def processed_file_names(self) -> str:
        name = '_'.join([cat[:2] for cat in self.categories])
        return f'{name}.pt'

    def download(self):
        pass

    def process(self):
        data_list = []
        for cat in self.categories:
            paths = glob.glob(osp.join(self.raw_dir, f'{cat}*.tri'))
            paths = [path[:-4] for path in paths]
            paths = sorted(paths, key=lambda e: (len(e), e))

            for path in paths:
                pos = read_txt_array(f'{path}.vert')
                face = read_txt_array(f'{path}.tri', dtype=torch.long)
                face = face - face.min()  # Ensure zero-based index.
                data = Data(pos=pos, face=face.t().contiguous())
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

class TOSCA(TOSCA_PT):
    def __init__(
        self,
        root: str,
        train: bool = True,
        pre_transform: Optional[Callable] = None,
        pre_transform_str="",
        pre_filter: Optional[Callable] = None,
        train_frac: float = 0.75,
        skip_process: bool = False,
    ):
        self.pre_transform = pre_transform
        self.pre_transform_str = pre_transform_str
        self.train_frac = train_frac
        self.skip_process = skip_process
        super().__init__(root, pre_transform=pre_transform, pre_filter=pre_filter)

        path = self.processed_paths[train]
        if not self.skip_process:
            self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        name = "_".join([cat[:2] for cat in self.categories])
        base_paths = [f"{name}" + _ + ".pt" for _ in ["test", "train"]]
        return [self.pre_transform_str + _ for _ in base_paths]

    def _process_mesh(self, data, tform, dlist):
        aux_data = copy.deepcopy(data)
        if tform is not None:
            aux_data = tform(aux_data)
        dlist.append(aux_data)

    def process(self):
        if not self.skip_process:
            tr_dlist, ts_dlist = [], []

            for cat_id, cat in enumerate(self.categories):
                print(f"Category: {cat}")
                paths = glob.glob(osp.join(self.raw_dir, f"{cat}*.mat"))
                paths = [path[:-4] for path in paths]
                paths = sorted(paths, key=lambda e: (len(e), e))

                # Take the first (1 - self.train_frac)% meshes as test set
                tst_paths = paths[: int(len(paths) * (1 - self.train_frac))]
                for path in paths:
                    mesh = scipy.io.loadmat(f"{path}.mat")
                    pos = torch.tensor(np.concatenate([mesh['surface'][0][0][0], mesh['surface'][0][0][1], mesh['surface'][0][0][2]], axis = -1))
                    face = torch.tensor(mesh['surface'][0][0][3].astype(np.int32))
                    face = face - face.min()

                    data = Data(
                        pos=pos, face=face.t().contiguous(), y=torch.tensor(cat_id)
                    )
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if not (path in tst_paths):  # This is a training mesh
                        self._process_mesh(data, self.pre_transform, tr_dlist)
                    else:  # Mesh is in the test paths
                        self._process_mesh(data, self.pre_transform, ts_dlist)

            # Save paths = [test path, train path]
            for data_list, _path in zip([ts_dlist, tr_dlist], self.processed_paths):
                torch.save(self.collate(data_list), _path)

def construct_loaders(args):
    batch_size = args.batch_size
    pre_tform = T.Compose([compute_normals_edges_from_mesh,  compute_area_from_mesh, compute_triangle_info_from_mesh, compute_di_angle])
    path = exp_utils.get_dataset_path("tosca_direct")
    TOSCA.processed_dir = osp.join(path, "processed_direct")

    if args.verbose:
        print("Creating datasets")
    train_data = TOSCA(root=path, train=True, pre_transform=pre_tform)

    train_batch_size = 1 

    test_dataset = TOSCA(root=path, train=False, pre_transform=pre_tform)

    loaders_dict = {
        "train": DataLoader(train_data, batch_size=train_batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }

    target_dim = 9

    return None, loaders_dict, None, target_dim


if __name__ == "__main__":
    experiment = exp_utils.Experiment(
        task_name="tosca",
        task_type="classification",
        construct_loaders=construct_loaders,
    )
    parser = exp_utils.parse_arguments()
    args = exp_utils.run_parser(parser)
    experiment.main(args)
