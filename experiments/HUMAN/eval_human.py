import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


import argparse
import copy
import os.path as osp
import shutil
from typing import Callable, Optional
import trimesh
import exp_utils
import numpy as np
import torch
import torch_geometric.transforms as T
from exp_utils import core_prepare_batch
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch_geometric.data import DataLoader

from egnn.transform.permute_nodes import PermuteNodesHuman
from egnn.transform.scale_roto_translation_transformer import ScaleRotoTranslationTransformer
from experiments.exp_utils import set_seed
from egnn.transform.preprocess import *
from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def shuffle_backwards(shuffle):
    a = torch.arange(len(shuffle))
    a[shuffle] = torch.arange(len(shuffle))
    return a


class HUMAN(InMemoryDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        test_type="gauge_transformations",
        transform: Optional[Callable] = None,
        pre_transform_train: Optional[Callable] = None,
        pre_transform_test_gauge: Optional[Callable] = None,
        pre_transform_test_rt: Optional[Callable] = None,
        pre_transform_test_perm: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        train_frac: float = 0.75,
    ):
        self.root = root
        self.pre_transform_train = pre_transform_train
        self.pre_transform_test_gauge = pre_transform_test_gauge
        self.pre_transform_test_rt = pre_transform_test_rt
        self.pre_transform_test_perm = pre_transform_test_perm
        super().__init__(root, transform, pre_transform=None, pre_filter=pre_filter)
        if train:
            path = self.processed_paths[0]
        elif test_type == "train_transforms":
            path = self.processed_paths[1]
        elif test_type == "gauge_transforms":
            path = self.processed_paths[2]
        elif test_type == "roto_translation_transforms":
            path = self.processed_paths[3]
        elif test_type == "permutation_transforms":
            path = self.processed_paths[4]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        return [
            "train.pt",
            "test.pt",
            "test_gauge.pt",
            "test_rototranslation.pt",
            "test_permutations.pt",
        ]

    def _process_mesh(self, data, tform, dlist):
        aux_data = copy.deepcopy(data)
        if tform is not None:
            aux_data = tform(aux_data)
        dlist.append(aux_data)

    def read_seg(self, seg):
        seg_labels = np.loadtxt(open(seg, 'r'), dtype='long')
        return seg_labels

    def _process_mesh(self, data, tform, dlist):
        aux_data = copy.deepcopy(data)
        if tform is not None:
            aux_data = tform(aux_data)
        dlist.append(aux_data)

    def process(self):
        data_list = [[], [], [], [], []]

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


                if split == 'train':  # This is a training mesh
                    self._process_mesh(data, self.pre_transform_train, data_list[0])
                else:  # Mesh is in the test paths
                    # Append "untransformed" test mesh
                    self._process_mesh(data, self.pre_transform_train, data_list[1])

                    # Append gauge transformed test mesh
                    self._process_mesh(data, self.pre_transform_test_gauge, data_list[2])

                    # Append rototranslation transformed test mesh
                    self._process_mesh(data, self.pre_transform_test_rt, data_list[3])

                    # Append permutation transformed test mesh
                    self._process_mesh(data, self.pre_transform_test_perm, data_list[4])

        for dl, _path in zip(data_list, self.processed_paths):
            torch.save(self.collate(dl), _path)


def eval_model(args):
    set_seed(
        args.seed
    )  # Important to have same seed for input shuffle and target shuffle!
    # build test loaders
    pre_tform_train = [compute_area_from_mesh, compute_normals_edges_from_mesh, compute_triangle_info_from_mesh, compute_di_angle, compute_hks]
    pre_tform_test_gauge = [compute_area_from_mesh, compute_normals_edges_from_mesh, compute_triangle_info_from_mesh, compute_di_angle, compute_hks]
    pre_tform_test_rt = [
        ScaleRotoTranslationTransformer(translation_mag=100),
        compute_area_from_mesh,
        compute_normals_edges_from_mesh,
        compute_triangle_info_from_mesh,
        compute_di_angle,
        compute_hks
    ]


    permute_nodes = PermuteNodesHuman()
    pre_tform_test_perm = [permute_nodes, compute_area_from_mesh, compute_normals_edges_from_mesh, compute_triangle_info_from_mesh, compute_di_angle, compute_hks]

    pre_transform_dict = {
        "pre_transform_train": T.Compose(pre_tform_train),
        "pre_transform_test_gauge": T.Compose(pre_tform_test_gauge),
        "pre_transform_test_rt": T.Compose(pre_tform_test_rt),
        "pre_transform_test_perm": T.Compose(pre_tform_test_perm),
    }

    # Provide a path to load and store the dataset
    path = exp_utils.get_dataset_path("human_direct")

    train_dataset = HUMAN(
        path, train=True, test_type="train_transforms", **pre_transform_dict
    )
    test_dataset = HUMAN(
        path, train=False, test_type="train_transforms", **pre_transform_dict
    )
    test_gauge_dataset = HUMAN(
        path, train=False, test_type="gauge_transforms", **pre_transform_dict
    )
    test_rt_dataset = HUMAN(
        path, train=False, test_type="roto_translation_transforms", **pre_transform_dict
    )
    test_perm_dataset = HUMAN(
        path, train=False, test_type="permutation_transforms", **pre_transform_dict
    )
    
    loaders_dict = {
        "train": DataLoader(train_dataset, batch_size=1),
        "test": DataLoader(test_dataset, batch_size=1),
        "test_gauge": DataLoader(test_gauge_dataset, batch_size=1),
        "test_srt": DataLoader(test_rt_dataset, batch_size=1),
        "test_perm": DataLoader(test_perm_dataset, batch_size=1),
    }

    args.target_dim = 8


    model = torch.load(args.model_path, map_location=device)
    model.eval()

    criterion = torch.nn.NLLLoss()
    metrics_dict = {"nll": Loss(criterion), "accuracy": Accuracy()}
    eval_dict = {}
    # for eval_name in ["test", "test_gauge", "test_rt", "test_perm"]:
    for eval_name in ["test", "test_gauge", "test_srt","test_perm"]:
        target = None
        prepare_batch_fn = core_prepare_batch(target)
        eval_dict[eval_name] = create_supervised_evaluator(
            model,
            metrics=metrics_dict,
            device=device,
            prepare_batch=prepare_batch_fn,
        )
        eval_dict[eval_name].run(loaders_dict[eval_name])
        metrics = eval_dict[eval_name].state.metrics
        print(
            f"{eval_name.upper()} Results "
            f"Avg accuracy: {metrics['accuracy']:.5f} Avg loss: {metrics['nll']:.5f}"
        )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMAN evaluation parser")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--task_type", type=str, default="classification", help="provide model path")
    parser.add_argument("--model_path", type=str, help="provide model path")
    parser.add_argument("--num_nodes", default=6890, type=int, help="provide model path")
    parser.add_argument("-load_model", action="store_true")
    parser.add_argument("--model", default="GemCNN", type=str)
    args = parser.parse_args()

    eval_model(args)
