import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import copy
import os.path as osp
import shutil
from typing import Callable, Optional

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, extract_zip
from torch_geometric.io import read_ply

import exp_utils
from egnn.transform.preprocess import *


class FAUST(torch_geometric.datasets.FAUST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_transform_str="",
        pre_filter: Optional[Callable] = None,
        skip_process: bool = False,
    ):

        self.pre_transform = pre_transform
        self.pre_transform_str = pre_transform_str
        self.skip_process = skip_process
        super().__init__(
            root, transform, pre_transform=pre_transform, pre_filter=pre_filter
        )

        path = self.processed_paths[train]
        if not self.skip_process:
            self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        base_paths = ["test.pt", "training.pt"]
        return [self.pre_transform_str + _ for _ in base_paths]

    def _process_mesh(self, data, tform, dlist):
        aux_data = copy.deepcopy(data)
        if tform is not None:
            aux_data = tform(aux_data)
        dlist.append(aux_data)

    def process(self):
        if not self.skip_process:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)

            path = osp.join(self.raw_dir, "MPI-FAUST", "training", "registrations")
            path = osp.join(path, "tr_reg_{0:03d}.ply")

            tr_dlist, ts_dlist = [], []

            for mesh_ix in range(100):
                data = read_ply(path.format(mesh_ix))
                data.y = torch.tensor([mesh_ix % 10], dtype=torch.long)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if mesh_ix < 80:
                    self._process_mesh(data, self.pre_transform, tr_dlist)
                else:
                    self._process_mesh(data, self.pre_transform, ts_dlist)

            # Save paths = [test path, train path]
            for data_list, _path in zip([ts_dlist, tr_dlist], self.processed_paths):
                torch.save(self.collate(data_list), _path)

            shutil.rmtree(osp.join(self.raw_dir, "MPI-FAUST"))


def construct_loaders(args):
    batch_size = args.batch_size
    pre_tform = T.Compose([compute_normals_edges_from_mesh,  compute_area_from_mesh, compute_triangle_info_from_mesh, compute_di_angle])
    path = exp_utils.get_dataset_path("faust_direct")
    FAUST.processed_dir = osp.join(path, "processed_direct")


    if args.verbose:
        print("Creating datasets")
    train_data = FAUST(path, train=True, pre_transform=pre_tform)

    train_batch_size = 1
    num_nodes = train_data[0].num_nodes

    test_dataset = FAUST(path, train=False, pre_transform=pre_tform)

    loaders_dict = {
        "train": DataLoader(train_data, batch_size=train_batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }

    target_dim = num_nodes

    return None, loaders_dict, num_nodes, target_dim


if __name__ == "__main__":
    experiment = exp_utils.Experiment(
        task_name="faust", task_type="segmentation", construct_loaders=construct_loaders
    )
    parser = exp_utils.parse_arguments()
    args = exp_utils.run_parser(parser)
    experiment.main(args)
