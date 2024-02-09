# E(3)-Equivariant Mesh Neural Networks

https://arxiv.org/pdf/2402.04821.pdf

Published at AISTATS 2024

## Running experiments

Run this command

```
# Train
sh scripts/{data}/train.sh
# Eval
sh scripts/{data}/eval.sh
```

## Environment setups

Follow these commands below:
```
conda create --name emnn python=3.7
conda activate emnn

conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch

conda install pyg=2.0.3 -c pyg
pip install wandb pytorch-ignite openmesh opt_einsum trimesh
```
