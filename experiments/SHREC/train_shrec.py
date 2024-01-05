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
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader

from egnn.transform.preprocess import *

import os

import torch

from torch_geometric.data import (Data, InMemoryDataset)
from typing import Callable, List, Optional
import trimesh

class SHREC(InMemoryDataset):

    def __init__(
        self,
        root: str,
        train: bool,
        pre_transform_str="",
        categories: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        skip_process: bool = False,
    ):
        categories = os.listdir(osp.join(root, 'raw'))
        self.categories = categories
        self.pre_transform_str = pre_transform_str
        self.skip_process = skip_process
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[not(train)]
        if not self.skip_process:
            self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['cat0.vert', 'cat0.tri']
    
    @property
    def processed_file_names(self):
        base_paths = [_ + ".pt" for _ in ["train", "test"]]
        return [self.pre_transform_str + _ for _ in base_paths]
    
    def download(self):
        pass

    def process(self):
        data_list = [[], []]
        train_paths = []
        test_paths = []
        for cat in self.categories:
            train_paths.extend(glob.glob(osp.join(self.raw_dir, f'{cat}/train/*.obj')))
            test_paths.extend(glob.glob(osp.join(self.raw_dir, f'{cat}/test/*.obj')))
            train_paths = sorted(train_paths, key=lambda e: (len(e), e))
            test_paths = sorted(test_paths, key=lambda e: (len(e), e))

        for i, paths in enumerate([train_paths, test_paths]):
            for path in paths:
                mesh = trimesh.load(path, force='mesh')
                pos = torch.tensor(mesh.vertices)
                face = torch.tensor(mesh.faces)
                face = face - face.min()  # Ensure zero-based index.
                data = Data(pos=pos, face=face.T.contiguous(), y=torch.tensor(self.categories.index(path.split('/')[3])))
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list[i].append(data)


        for dl, _path in zip(data_list, self.processed_paths):
            torch.save(self.collate(dl), _path)

def construct_loaders(args):
    batch_size = args.batch_size
    pre_tform = T.Compose([compute_normals_edges_from_mesh,  compute_area_from_mesh, compute_di_angle, compute_hks])
    path = exp_utils.get_dataset_path(args.shrec)
    SHREC.processed_dir = osp.join(path, "processed_direct")

    # --------------- Load datasets ---------------
    if args.verbose:
        print("Creating datasets")
    train_data = SHREC(root=path, train=True, pre_transform=pre_tform)

    # Scale batch_size according to train fraction
    train_batch_size = 1

    # Evaluation takes place on full test set
    # Unseen and NOT-transformed meshes
    test_dataset = SHREC(root=path, train=False, pre_transform=pre_tform)

    # import pdb; pdb.set_trace()
    loaders_dict = {
        "train": DataLoader(train_data, batch_size=train_batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }

    # There are 30 classes in this dataset
    target_dim = 30

    return None, loaders_dict, None, target_dim


if __name__ == "__main__":
    experiment = exp_utils.Experiment(
        task_name="shrec",
        task_type="classification",
        construct_loaders=construct_loaders,
    )
    parser = exp_utils.parse_arguments()
    args = exp_utils.run_parser(parser)
    experiment.main(args)
