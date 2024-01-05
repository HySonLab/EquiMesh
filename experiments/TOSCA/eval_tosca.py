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
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.data import Data
from exp_utils import core_prepare_batch
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

import egnn.models as models
from egnn.transform.scale_roto_translation_transformer import ScaleRotoTranslationTransformer
from experiments.exp_utils import set_seed
from egnn.transform.preprocess import *
from train_tosca import TOSCA_PT
import scipy.io
import glob 


device = "cuda" if torch.cuda.is_available() else "cpu"

class TOSCA(TOSCA_PT):
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
        self.train_frac = train_frac
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
                    pos=pos.float(), face=face.t().contiguous(), y=torch.tensor(cat_id)
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if not (path in tst_paths):  # This is a training mesh
                    self._process_mesh(data, self.pre_transform_train, tr_dlist)
                else:  # Mesh is in the test paths
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

def eval_model(args):
    set_seed(
        args.seed
    )  # Important to have same seed for input shuffle and target shuffle!
    # build test loaders
    pre_tform_train = [compute_area_from_mesh, compute_normals_edges_from_mesh, compute_triangle_info_from_mesh, compute_di_angle]
    pre_tform_test_gauge = [
        compute_area_from_mesh,
        compute_normals_edges_from_mesh,
        compute_triangle_info_from_mesh,
        compute_di_angle
    ]
    pre_tform_test_rt = [
        ScaleRotoTranslationTransformer(translation_mag=100),
        compute_area_from_mesh,
        compute_normals_edges_from_mesh,
        compute_triangle_info_from_mesh,
        compute_di_angle
    ]

    pre_transform_dict = {
        "pre_transform_train": T.Compose(pre_tform_train),
        "pre_transform_test_gauge": T.Compose(pre_tform_test_gauge),
        "pre_transform_test_rt": T.Compose(pre_tform_test_rt),
    }

    # Provide a path to load and store the dataset
    path = exp_utils.get_dataset_path("tosca_direct")

    train_dataset = TOSCA(
        path, train=True, test_type="train_transforms", **pre_transform_dict
    )
    test_dataset = TOSCA(
        path, train=False, test_type="train_transforms", **pre_transform_dict
    )
    test_gauge_dataset = TOSCA(
        path, train=False, test_type="gauge_transforms", **pre_transform_dict
    )
    test_rt_dataset = TOSCA(
        path, train=False, test_type="roto_translation_transforms", **pre_transform_dict
    )
    
    loaders_dict = {
        "train": DataLoader(train_dataset, batch_size=1),
        "test": DataLoader(test_dataset, batch_size=1),
        "test_gauge": DataLoader(test_gauge_dataset, batch_size=1),
        "test_srt": DataLoader(test_rt_dataset, batch_size=1),
    }

    args.target_dim = 9

    model = torch.load(args.model_path, map_location=device)
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
    parser.add_argument("--num_nodes", default=6890, type=int, help="provide model path")
    parser.add_argument("-load_model", action="store_true")

    parser.add_argument("--model", default="GemCNN", type=str)

    args = parser.parse_args()

    args.mean_pooling = True
    args.hks = False
    args.hidden_dim = 16

    eval_model(args)
