import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import copy
import os.path as osp
from typing import Callable, Optional
import numpy as np
import torch
import torch_geometric.transforms as T
import exp_utils
from egnn.transform.preprocess import *
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import trimesh

class HUMAN(InMemoryDataset):
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
        self.root = root
        self.pre_transform = pre_transform
        self.pre_transform_str = pre_transform_str
        self.skip_process = skip_process

        super().__init__(
            root, transform, pre_transform=pre_transform, pre_filter=pre_filter
        )

        path = self.processed_paths[not(train)]
        if not self.skip_process:
            self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        base_paths = ["train.pt", "test.pt"]
        return [self.pre_transform_str + _ for _ in base_paths]

    def _process_mesh(self, data, tform, dlist):
        aux_data = copy.deepcopy(data)
        if tform is not None:
            aux_data = tform(aux_data)
        dlist.append(aux_data)

    def read_seg(self, seg):
        seg_labels = np.loadtxt(open(seg, 'r'), dtype='long')
        return seg_labels

    def process(self):
        data_list = [[], []]

        for i, split in enumerate(['train', 'test']):
            for path in os.listdir(os.path.join(self.root, f'raw/{split}')):
                mesh = trimesh.load(os.path.join(self.root, f'raw/{split}', path))
                pos = torch.tensor(mesh.vertices).float()
                face = torch.tensor(mesh.faces)
                face = face - face.min()  # Ensure zero-based index.
                
                label = self.read_seg(os.path.join(self.root, f'raw/seg/', path).replace('.ply', '.eseg'))

                data = Data(pos=pos, face=face.T.contiguous(), y=torch.tensor(label, dtype=torch.long))
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list[i].append(data)

        for dl, _path in zip(data_list, self.processed_paths):
            torch.save(self.collate(dl), _path)

def construct_loaders(args):
    batch_size = args.batch_size
    pre_tform = T.Compose([compute_normals_edges_from_mesh,  compute_area_from_mesh, compute_triangle_info_from_mesh, compute_di_angle, compute_hks, compute_hierarchy])
    path = exp_utils.get_dataset_path("human_direct")
    HUMAN.processed_dir = osp.join(path, "processed_direct")

    # --------------- Load datasets ---------------
    if args.verbose:
        print("Creating datasets")
    train_data = HUMAN(path, train=True, pre_transform=pre_tform)
    train_batch_size = 1
    test_dataset = HUMAN(path, train=False, pre_transform=pre_tform)
    loaders_dict = {
        "train": DataLoader(train_data, batch_size=train_batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }
    target_dim = 8

    return None, loaders_dict, None, target_dim


if __name__ == "__main__":
    experiment = exp_utils.Experiment(
        task_name="human", task_type="classification", construct_loaders=construct_loaders
    )
    parser = exp_utils.parse_arguments()
    args = exp_utils.run_parser(parser)
    experiment.main(args)
