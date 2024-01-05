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
import torch_geometric.transforms as T
from exp_utils import core_prepare_batch
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from egnn.transform.scale_roto_translation_transformer import ScaleRotoTranslationTransformer

from experiments.exp_utils import set_seed
from egnn.transform.preprocess import *
import glob 

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    DataLoader
)

device = "cuda" if torch.cuda.is_available() else "cpu"

import trimesh

def shuffle_backwards(shuffle):
    a = torch.arange(len(shuffle))
    a[shuffle] = torch.arange(len(shuffle))
    return a


class SHREC(InMemoryDataset):
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
        categories = os.listdir(osp.join(root, 'raw'))
        self.categories = categories
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
        tr_dlist, ts_dlist, ts_g_dlist, ts_rt_dlist, ts_perm_dlist = [], [], [], [], []

        data_list = [[], [], [], [], []]
        processed_data_list = [[], [], [], [], []]
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
                data = Data(pos=pos.float(), face=face.T.contiguous(), y=torch.tensor(self.categories.index(path.split('/')[3])))
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list[i].append(data)
        
                if i == 0:  # This is a training mesh
                    self._process_mesh(data, self.pre_transform_train, processed_data_list[0])
                else:  # Mesh is in the test paths
                    # Append "untransformed" test mesh
                    self._process_mesh(data, self.pre_transform_train, processed_data_list[1])

                    # Append gauge transformed test mesh
                    self._process_mesh(data, self.pre_transform_test_gauge, processed_data_list[2])

                    # Append rototranslation transformed test mesh
                    self._process_mesh(data, self.pre_transform_test_rt, processed_data_list[3])

                    # Append permutation transformed test mesh
                    self._process_mesh(data, self.pre_transform_test_perm, processed_data_list[4])

        for data_list_, _path in zip(
            processed_data_list,
            self.processed_paths,
        ):
            print(len(data_list_))
            torch.save(self.collate(data_list_), _path)


def eval_model(args):
    set_seed(
        args.seed
    )  # Important to have same seed for input shuffle and target shuffle!
    # build test loaders
    pre_tform_train = [compute_area_from_mesh, compute_normals_edges_from_mesh, compute_triangle_info_from_mesh, compute_di_angle, compute_hks]
    pre_tform_test_gauge = [
        compute_area_from_mesh,
        compute_normals_edges_from_mesh,
        compute_triangle_info_from_mesh,
        compute_di_angle,
        compute_hks
    ]
    pre_tform_test_srt = [
        ScaleRotoTranslationTransformer(translation_mag=100),
        compute_area_from_mesh,
        compute_normals_edges_from_mesh,
        compute_triangle_info_from_mesh,
        compute_di_angle,
        compute_hks
    ]

    pre_transform_dict = {
        "pre_transform_train": T.Compose(pre_tform_train),
        "pre_transform_test_gauge": T.Compose(pre_tform_test_gauge),
        "pre_transform_test_rt": T.Compose(pre_tform_test_srt),
    }

    # Provide a path to load and store the dataset
    path = exp_utils.get_dataset_path(args.shrec)

    train_dataset = SHREC(
        path, train=True, test_type="train_transforms", **pre_transform_dict
    )
    test_dataset = SHREC(
        path, train=False, test_type="train_transforms", **pre_transform_dict
    )
    test_gauge_dataset = SHREC(
        path, train=False, test_type="gauge_transforms", **pre_transform_dict
    )
    test_srt_dataset = SHREC(
        path, train=False, test_type="roto_translation_transforms", **pre_transform_dict
    )
    
    loaders_dict = {
        "train": DataLoader(train_dataset, batch_size=1),
        "test": DataLoader(test_dataset, batch_size=1),
        "test_gauge": DataLoader(test_gauge_dataset, batch_size=1),
        "test_srt": DataLoader(test_srt_dataset, batch_size=1),
    }


    model = torch.load(args.model_path, map_location=device)

    # # if spiralnet, permute model.indices using shuffle
    # [2, 0, 1, 3, 4]

    # #points x #spiral_length
    # row 0 == [1, 2, 3] -> row 2 == [0, 1, 3]

    # eval on desired dataset
    model.eval()
    criterion = torch.nn.NLLLoss()
    metrics_dict = {"nll": Loss(criterion), "accuracy": Accuracy()}
    eval_dict = {}
    # for eval_name in ["test", "test_gauge", "test_rt", "test_perm"]:
    for eval_name in ["test", "test_gauge", "test_srt"]:
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
    parser.add_argument(
        "--task_type", type=str, default="classification", help="provide model path"
    )
    parser.add_argument("--model_path", type=str, help="provide model path")
    parser.add_argument(
        "--num_nodes", default=6890, type=int, help="provide model path"
    )
    parser.add_argument("-load_model", action="store_true")

    parser.add_argument("--model", default="GemCNN", type=str)

    parser.add_argument("-shrec", default="", type=str)

    args = parser.parse_args()
    eval_model(args)
