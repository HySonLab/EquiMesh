import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import argparse
import json
import os.path as osp
import random

import numpy as np
import torch
import wandb
import yaml
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss, RunningAverage


import egnn.models as models


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_dataset_path(task_name: str) -> str:
    # Retrieve path for given task_name

    with open(osp.join(osp.dirname(__file__), "paths.json")) as f:
        all_paths = json.load(f)
        path = all_paths[task_name]

    return path

def construct_target(task_type="segmentation", args=None, device=None):
    if task_type == "segmentation":
        aux = torch.arange(args.num_nodes, dtype=torch.long, device=device)
        return aux.expand(args.batch_size, args.num_nodes).flatten()
    if task_type == "classification":
        return None

    raise ValueError("Task type {} not understood".format(task_type))

def core_prepare_batch(target=None, shuffle=None):
    if target is None:

        def prepare_batch(batch, device, non_blocking=False):
            data = batch.to(device)
            data.pos = data.pos.float()
            if shuffle is None:
                data.pos = data.pos.float()
            else:
                data.pos = data.pos.float()[shuffle]
            return data, data.y.to(device)

    else:

        def prepare_batch(batch, device, non_blocking=False):
            data = batch.to(device)
            if shuffle is None:
                data.pos = data.pos.float()
            else:
                data.pos = data.pos.float()[shuffle]
            return data, target.to(device)

    return prepare_batch

class GEMEngine:
    def __init__(
        self, model, optimizer, criterion, device, prepare_batch, grad_accum_steps=1
    ):
    
        self.trainer = create_supervised_trainer(
            model,
            optimizer,
            criterion,
            device=device,
            prepare_batch=prepare_batch,
            gradient_accumulation_steps=grad_accum_steps,
        )

        RunningAverage(output_transform=lambda x: x).attach(self.trainer, "loss")
        ProgressBar().attach(self.trainer, ["loss"])

        metrics_dict = {"nll": Loss(criterion), "accuracy": Accuracy()}

        self.eval_dict = {}
        for eval_name in ["train", "test", "test_tf"]:
            self.eval_dict[eval_name] = create_supervised_evaluator(
                model, metrics=metrics_dict, device=device, prepare_batch=prepare_batch
            )
        
        # Initial best result
        self.trainer.best_train_result = 0
        self.trainer.best_test_result = 0
        self.model = model

    def set_epoch_loggers(self, loaders_dict):
        def inner_log(engine, tag, model):
            self.eval_dict[tag].run(loaders_dict[tag])
            metrics = self.eval_dict[tag].state.metrics

            if engine.save_model:
                if tag == 'train':
                    self.trainer.temp_train_result = metrics['accuracy']
                elif tag == 'test':
                    if self.trainer.temp_train_result  > self.trainer.best_train_result: 
                        self.trainer.best_train_result = self.trainer.temp_train_result
                        self.trainer.best_test_result = metrics['accuracy']
                        print(f'Save best model with accuracy: {self.trainer.best_train_result}')
                        torch.save(model, engine.file_path)
                    elif self.trainer.best_train_result == self.trainer.temp_train_result:
                        if metrics['accuracy'] > self.trainer.best_test_result:
                            self.trainer.best_test_result = metrics['accuracy']
                            print(f'Save best model with accuracy: {self.trainer.best_train_result}')
                            torch.save(model, engine.file_path)
                        

            print(
                f"{tag.upper()} Results - Epoch: {engine.state.epoch} "
                f"Avg accuracy: {metrics['accuracy']:.5f} Avg loss: {metrics['nll']:.5f}"
            )

        # These logs are created regardless of the wandb choice
        if loaders_dict["train"] is not None:

            @self.trainer.on(Events.EPOCH_COMPLETED)
            def log_train_results(engine):
                inner_log(engine, "train", self.model)

        if loaders_dict["test"] is not None:

            @self.trainer.on(Events.EPOCH_COMPLETED)
            def log_test_results(engine):
                inner_log(engine, "test", self.model)

class Experiment:
    def __init__(self, task_name, task_type, construct_loaders):
        self.task_name = task_name
        self.task_type = task_type
        self.construct_loaders = construct_loaders

    def main(self, args):

        set_seed(args.seed)

        # --------------- Dataset ---------------
        (
            transform,
            loaders_dict,
            args.num_nodes,
            args.target_dim,
        ) = self.construct_loaders(args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.task_type == "classification":
            args.model_batch = None
        if self.task_type == "segmentation":
            args.model_batch = 100_000

        if args.verbose:
            print(f"Building model of type: {args.model}")

        model = models.__dict__[args.model](args).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.NLLLoss()

        # --------------- Train ---------------
        target = construct_target(self.task_type, args, device)
        prepare_batch_fn = core_prepare_batch(target)

        engine = GEMEngine(
            model,
            optimizer,
            criterion,
            device,
            prepare_batch_fn,
            grad_accum_steps=args.grad_accum_steps,
        )
        engine.set_epoch_loggers(loaders_dict)

        engine.trainer.save_model = args.save_model
        folder_path = args.exp_name
        file_path = f"{folder_path}/trained_model_best.h5"
        engine.trainer.file_path = file_path
        if args.save_model:
            os.makedirs(folder_path, exist_ok=True)


        engine.trainer.run(loaders_dict["train"], max_epochs=args.epochs)

def parse_arguments():
    parser = argparse.ArgumentParser(description="EMAN Parser")

    parser.add_argument("-yaml", "--yaml_file", default="", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--model", default="GemCNN", type=str)
    parser.add_argument("--exp_name", default='model', type=str)
    parser.add_argument("--layer", default='', type=str)
    parser.add_argument("--hier", action="store_true")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate")
    parser.add_argument(
        "-gas",
        "--grad_accum_steps",
        default=1,
        type=int,
        help="apply optimizer step every several minibatches",
    )
    parser.add_argument(
        "-bs", "--batch_size", default=1, type=int, help="Number of meshes per batch"
    )
    parser.add_argument("-save_model", action="store_true")
    parser.add_argument("-save_model_dict", action="store_true")

    parser.add_argument("-shrec", default="", type=str)

    return parser

def run_parser(parser):
    args = parser.parse_args()
    if args.yaml_file != "":
        opt = yaml.load(open(args.yaml_file), Loader=yaml.FullLoader)
        args.__dict__.update(opt)
    return args
