import numpy as np
import torch
from torch_geometric.data import Data


class PermuteNodes:
    def __init__(self, shuffle):
        self.shuffle = shuffle

    def shuffle_backwards(self):
        a = torch.arange(len(self.shuffle))
        a[self.shuffle] = torch.arange(len(self.shuffle))
        return a

    def __call__(self, data):
        pos = data.pos
        face = data.face

        face_indices = torch.arange(len(pos))  # shuffling used for computing faces

        backward_shuffle = self.shuffle_backwards()
        pos_shuffle = pos.clone()[backward_shuffle]
        face_indices_shuffle = face_indices.clone()[self.shuffle]

        face_shuffle = face.clone()
        face_shuffle[0], face_shuffle[1], face_shuffle[2] = (
            face_indices_shuffle.clone()[face_shuffle[0]],
            face_indices_shuffle.clone()[face_shuffle[1]],
            face_indices_shuffle.clone()[face_shuffle[2]],
        )

        data_shuffle = data.clone()
        data_shuffle.pos = pos_shuffle
        data_shuffle.face = face_shuffle

        return data_shuffle
    
class PermuteNodesHuman:
    def __init__(self):
        pass

    def shuffle_backwards(self, shuffle):
        a = torch.arange(len(shuffle))
        a[shuffle] = torch.arange(len(shuffle))
        return a

    def __call__(self, data):
        pos = data.pos
        face = data.face
        face_indices = torch.arange(len(pos))  # shuffling used for computing faces

        shuffle = torch.tensor(np.random.choice(pos.shape[0], size=pos.shape[0], replace=False))
        backward_shuffle = self.shuffle_backwards(shuffle)
        pos_shuffle = pos.clone()[backward_shuffle]
        face_indices_shuffle = face_indices.clone()[shuffle]

        face_shuffle = face.clone()
        face_shuffle[0], face_shuffle[1], face_shuffle[2] = (
            face_indices_shuffle.clone()[face_shuffle[0]],
            face_indices_shuffle.clone()[face_shuffle[1]],
            face_indices_shuffle.clone()[face_shuffle[2]],
        )

        data_shuffle = data.clone()
        data_shuffle.pos = pos_shuffle
        data_shuffle.face = face_shuffle


        return data_shuffle

