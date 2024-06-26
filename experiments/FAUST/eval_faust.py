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

import exp_utils
import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from exp_utils import core_prepare_batch
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch_geometric.data import DataLoader, extract_zip
from torch_geometric.io import read_ply

from egnn.transform.permute_nodes import PermuteNodes
from egnn.transform.scale_roto_translation_transformer import ScaleRotoTranslationTransformer
from experiments.exp_utils import set_seed
from egnn.transform.preprocess import *


device = "cuda" if torch.cuda.is_available() else "cpu"


def shuffle_backwards(shuffle):
    a = torch.arange(len(shuffle))
    a[shuffle] = torch.arange(len(shuffle))
    return a


class FAUST(torch_geometric.datasets.FAUST):
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
    ):

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

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        path = osp.join(self.raw_dir, "MPI-FAUST", "training", "registrations")
        path = osp.join(path, "tr_reg_{0:03d}.ply")

        tr_dlist, ts_dlist, ts_g_dlist, ts_rt_dlist, ts_perm_dlist = [], [], [], [], []

        for mesh_ix in range(100):
            data = read_ply(path.format(mesh_ix))
            data.y = torch.tensor([mesh_ix % 10], dtype=torch.long)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if mesh_ix < 80:
                self._process_mesh(data, self.pre_transform_train, tr_dlist)
            else:
                # Append "untransformed" test mesh
                self._process_mesh(data, self.pre_transform_train, ts_dlist)

                # Append gauge transformed test mesh
                self._process_mesh(data, self.pre_transform_test_gauge, ts_g_dlist)

                # Append rototranslation transformed test mesh
                self._process_mesh(data, self.pre_transform_test_rt, ts_rt_dlist)

                # Append permutation transformed test mesh
                self._process_mesh(data, self.pre_transform_test_perm, ts_perm_dlist)

        for data_list, _path in zip(
            [tr_dlist, ts_dlist, ts_g_dlist, ts_rt_dlist, ts_perm_dlist],
            self.processed_paths,
        ):
            torch.save(self.collate(data_list), _path)

        shutil.rmtree(osp.join(self.raw_dir, "MPI-FAUST"))


def eval_model(args):
    set_seed(
        args.seed
    )  # Important to have same seed for input shuffle and target shuffle!
    # build test loaders
    pre_tform_train = [compute_normals_edges_from_mesh,  compute_area_from_mesh, compute_triangle_info_from_mesh, compute_di_angle]
    pre_tform_test_gauge = [compute_normals_edges_from_mesh,  compute_area_from_mesh, compute_triangle_info_from_mesh, compute_di_angle]
    pre_tform_test_rt = [ScaleRotoTranslationTransformer(translation_mag=100), compute_normals_edges_from_mesh,  compute_area_from_mesh, compute_triangle_info_from_mesh, compute_di_angle]

    shuffle = torch.tensor(np.random.choice(6890, size=6890, replace=False))
    backward_shuffle = shuffle_backwards(shuffle)
    permute_nodes = PermuteNodes(shuffle)
    pre_tform_test_perm = [permute_nodes, compute_normals_edges_from_mesh,  compute_area_from_mesh, compute_triangle_info_from_mesh, compute_di_angle]

    pre_transform_dict = {
        "pre_transform_train": T.Compose(pre_tform_train),
        "pre_transform_test_gauge": T.Compose(pre_tform_test_gauge),
        "pre_transform_test_rt": T.Compose(pre_tform_test_rt),
        "pre_transform_test_perm": T.Compose(pre_tform_test_perm),
    }

    # Provide a path to load and store the dataset.
    path = exp_utils.get_dataset_path("faust_direct")
    train_dataset = FAUST(
        path, train=True, test_type="train_transforms", **pre_transform_dict
    )
    test_dataset = FAUST(
        path, train=False, test_type="train_transforms", **pre_transform_dict
    )
    test_gauge_dataset = FAUST(
        path, train=False, test_type="gauge_transforms", **pre_transform_dict
    )
    test_rt_dataset = FAUST(
        path, train=False, test_type="roto_translation_transforms", **pre_transform_dict
    )
    test_perm_dataset = FAUST(
        path, train=False, test_type="permutation_transforms", **pre_transform_dict
    )

    loaders_dict = {
        "train": DataLoader(train_dataset, batch_size=1),
        "test": DataLoader(test_dataset, batch_size=1),
        "test_gauge": DataLoader(test_gauge_dataset, batch_size=1),
        "test_rt": DataLoader(test_rt_dataset, batch_size=1),
        "test_perm": DataLoader(test_perm_dataset, batch_size=1),
    }


    args.model_batch = 100_000
    args.target_dim = args.num_nodes

    model = torch.load(args.model_path, map_location=device)


    model.eval()
    criterion = torch.nn.NLLLoss()
    metrics_dict = {"nll": Loss(criterion), "accuracy": Accuracy()}
    eval_dict = {}
    for eval_name in ["test", "test_gauge", "test_rt", "test_perm"]:
        if eval_name == "test_perm":
            target = torch.arange(args.num_nodes, dtype=torch.long, device=device)[backward_shuffle]

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

        else:
            target = torch.arange(args.num_nodes, dtype=torch.long, device=device)
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
    parser.add_argument(
        "--task_type", type=str, default="segmentation", help="provide model path"
    )
    parser.add_argument("--model_path", type=str, help="provide model path")
    parser.add_argument(
        "--num_nodes", default=6890, type=int, help="provide model path"
    )
    parser.add_argument("-load_model", action="store_true")
    parser.add_argument("-load_model_dict", action="store_true")

    parser.add_argument("--model", default="GemCNN", type=str)



    args = parser.parse_args()

    eval_model(args)
