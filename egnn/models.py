import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool, LayerNorm

from egnn.nn.egnn_conv import *
from .aggregator import *
from .utils import *

class TOSCA(torch.nn.Module):
    def __init__(self, args):
        super(TOSCA, self).__init__()

        input_dim = 1    
        width = 16
        num_vector = 2

        if 'mc' in args.layer: 
            self.mc = True 
        else:
            self.mc = False

        self.feat = nn.Linear(input_dim, width)
        
        if args.layer == "EGNN":
            self.conv1 = EGNN(width,   width*2, width,   edges_in_d=1, coords_agg='mean')
            self.conv2 = EGNN(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")  
            self.conv3 = EGNN(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean") 
        elif args.layer == "EGNNmc":
            self.conv1 = EGNNmc(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
            self.conv2 = EGNNmc(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
            self.conv3 = EGNNmc(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  ) 
        elif args.layer == "MeshEGNN":
            self.conv1 = MeshEGNN(width,   width*2, width,   edges_in_d=1, coords_agg='mean')
            self.conv2 = MeshEGNN(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")  
            self.conv3 = MeshEGNN(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean") 
        elif args.layer == "MeshEGNNmc":
            self.conv1 = MeshEGNNmc(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
            self.conv2 = MeshEGNNmc(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
            self.conv3 = MeshEGNNmc(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  ) 

        self.lin1 = nn.Linear(width*8, width*8)
        self.lin2 = nn.Linear(width*8, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling

    def coord2area(self, face, coord):
        vec1 = coord[face[1]] - coord[face[0]]
        vec2 = coord[face[2]] - coord[face[0]]
        face_norm = vec1.cross(vec2)
        face_area = torch.norm(face_norm, p = 2, dim = -1) / 2 
        return face_area

    def pos_normalize(self, pos, batch):
        centroid = aggregate_mean(pos, batch, batch.max()+1)[batch]
        pos = pos - centroid
        m = aggregate_max(torch.sqrt(torch.sum(pos**2, axis=1)), batch, batch.max()+1)[batch]
        pos = pos / m.unsqueeze(-1)
        return pos

    def process_vertex2face(self, vertex2face, index_vertex, index_face, batch):
        index_batch = torch.arange(len(batch))
        batch = torch.repeat_interleave(index_batch.type_as(batch), batch, dim=0)
        vertex = vertex2face[..., 0]
        face = vertex2face[..., 1]
        vertex += index_vertex[:-1][batch]
        vertex = vertex.flatten()
        face += torch.cumsum(torch.cat([torch.zeros(1).type_as(index_face), index_face]), dim=0)[:-1][batch]
        face = face.flatten()
        return torch.stack([vertex, face]).T

    def forward(self, data):
        batch = data.batch
        pos = data.pos
        face = data.face.long()
        edge_attr = data.weight
        vertex2face = self.process_vertex2face(data.vertex2face, data.ptr, data.face_len, data.vertex2face_len)

        pos = self.pos_normalize(pos, batch)
        pos_origin = pos.clone()
        
        if self.mc:
            normal = data.normal
            normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
            normal = normal - normal.mean(dim = 0)
            pos = torch.cat([pos.unsqueeze(-1), normal.unsqueeze(-1)], dim=-1).float()

        area = scatter(self.coord2area(face, pos_origin)[vertex2face[:, 1]], vertex2face[:, 0], dim_size = data.pos.shape[0], reduce='mean' )
        x = self.feat(area.unsqueeze(-1))

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = edge_attr, face=face)
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = edge_attr, face=face)
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = edge_attr, face=face)

        x = F.relu(self.lin1(x))
        x = global_mean_pool(x, batch)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1) 

class FAUST(torch.nn.Module):
    def __init__(self, args):
        super(FAUST, self).__init__()

        input_dim = 1  
        width = 8
        num_vector = 2

        if 'mc' in args.layer: 
            self.mc = True 
        else:
            self.mc = False

        self.feat = nn.Linear(input_dim, width)
        
        if args.layer == "EGNN":
            self.conv1 = EGNN(width,   width*2, width,   edges_in_d=1, coords_agg='mean')
            self.conv2 = EGNN(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")  
            self.conv3 = EGNN(width*4, width*2, width*4, edges_in_d=1, coords_agg="mean") 
        elif args.layer == "EGNNmc":
            self.conv1 = EGNNmc(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
            self.conv2 = EGNNmc(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
            self.conv3 = EGNNmc(width*4, width*2, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  ) 
        elif args.layer == "MeshEGNN":
            self.conv1 = MeshEGNN(width,   width*2, width,   edges_in_d=1, coords_agg='mean')
            self.conv2 = MeshEGNN(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")  
            self.conv3 = MeshEGNN(width*4, width*2, width*4, edges_in_d=1, coords_agg="mean") 
        elif args.layer == "MeshEGNNmc":
            self.conv1 = MeshEGNNmc(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
            self.conv2 = MeshEGNNmc(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
            self.conv3 = MeshEGNNmc(width*4, width*2, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  ) 

        self.lin1 = nn.Linear(width*2, width*2)
        self.ln1 = LayerNorm(width*2)

    def coord2area(self, face, coord):
        vec1 = coord[face[1]] - coord[face[0]]
        vec2 = coord[face[2]] - coord[face[0]]
        face_norm = vec1.cross(vec2)
        face_area = torch.norm(face_norm, p = 2, dim = -1) / 2 
        return face_area

    def pos_normalize(self, pos, batch):
        centroid = aggregate_mean(pos, batch, batch.max()+1)[batch]
        pos = pos - centroid
        m = aggregate_max(torch.sqrt(torch.sum(pos**2, axis=1)), batch, batch.max()+1)[batch]
        pos = pos / m.unsqueeze(-1)
        return pos

    def process_vertex2face(self, vertex2face, index_vertex, index_face, batch):
        index_batch = torch.arange(len(batch))
        batch = torch.repeat_interleave(index_batch.type_as(batch), batch, dim=0)
        vertex = vertex2face[..., 0]
        face = vertex2face[..., 1]
        vertex += index_vertex[:-1][batch]
        vertex = vertex.flatten()
        face += torch.cumsum(torch.cat([torch.zeros(1).type_as(index_face), index_face]), dim=0)[:-1][batch]
        face = face.flatten()
        return torch.stack([vertex, face]).T

    def shuffle_backwards(self, shuffle):
        a = torch.arange(len(shuffle))
        a[shuffle] = torch.arange(len(shuffle))
        return a

    def forward(self, data):
        batch = data.batch
        pos = data.pos
        face = data.face.long()
        edge_attr = data.weight
        vertex2face = self.process_vertex2face(data.vertex2face, data.ptr, data.face_len, data.vertex2face_len)

        pos = self.pos_normalize(pos, batch)
        pos_origin = pos.clone()
        
        if self.mc:
            normal = data.normal
            normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
            normal = normal - normal.mean(dim = 0)
            pos = torch.cat([pos.unsqueeze(-1), normal.unsqueeze(-1)], dim=-1).float()

        area = scatter(self.coord2area(face, pos_origin)[vertex2face[:, 1]], vertex2face[:, 0], dim_size = data.pos.shape[0], reduce='mean' )
        x = self.feat(area.unsqueeze(-1))


        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = edge_attr, face=face)
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = edge_attr, face=face)
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = edge_attr, face=face)
        
        x = self.lin1(x)
        x = F.relu(self.ln1(x, batch))
        x = torch.matmul(x, x.T)

        if hasattr(data, "shuffle"):
            x = x[:, self.shuffle_backwards(data.shuffle)]

        return F.log_softmax(x, dim=1) 

class SHREC(torch.nn.Module):
    def __init__(self, args):
        super(SHREC, self).__init__()

        input_dim = 10  
        width = 32
        num_vector = 2

        if 'mc' in args.layer: 
            self.mc = True 
        else:
            self.mc = False

        self.feat = nn.Linear(input_dim, width)
        
        if args.layer == "EGNN":
            self.conv1 = EGNN(width,   width*2, width,   edges_in_d=1, coords_agg='mean')
            self.conv2 = EGNN(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")  
            self.conv3 = EGNN(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean") 
        elif args.layer == "EGNNmc":
            self.conv1 = EGNNmc(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
            self.conv2 = EGNNmc(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
            self.conv3 = EGNNmc(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  ) 
        elif args.layer == "MeshEGNN":
            self.conv1 = MeshEGNN(width,   width*2, width,   edges_in_d=1, coords_agg='mean')
            self.conv2 = MeshEGNN(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")  
            self.conv3 = MeshEGNN(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean") 
        elif args.layer == "MeshEGNNmc":
            self.conv1 = MeshEGNNmc(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
            self.conv2 = MeshEGNNmc(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
            self.conv3 = MeshEGNNmc(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  ) 

        self.lin1 = nn.Linear(width*8, width*8)
        self.ln1 = LayerNorm(width*8)
        self.lin2 = nn.Linear(width*8, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling

    def coord2area(self, face, coord):
        vec1 = coord[face[1]] - coord[face[0]]
        vec2 = coord[face[2]] - coord[face[0]]
        face_norm = vec1.cross(vec2)
        face_area = torch.norm(face_norm, p = 2, dim = -1) / 2 
        return face_area

    def pos_normalize(self, pos, batch):
        centroid = aggregate_mean(pos, batch, batch.max()+1)[batch]
        pos = pos - centroid
        m = aggregate_max(torch.sqrt(torch.sum(pos**2, axis=1)), batch, batch.max()+1)[batch]
        pos = pos / m.unsqueeze(-1)
        return pos

    def process_vertex2face(self, vertex2face, index_vertex, index_face, batch):
        index_batch = torch.arange(len(batch))
        batch = torch.repeat_interleave(index_batch.type_as(batch), batch, dim=0)
        vertex = vertex2face[..., 0]
        face = vertex2face[..., 1]
        vertex += index_vertex[:-1][batch]
        vertex = vertex.flatten()
        face += torch.cumsum(torch.cat([torch.zeros(1).type_as(index_face), index_face]), dim=0)[:-1][batch]
        face = face.flatten()
        return torch.stack([vertex, face]).T

    def forward(self, data):
        batch = data.batch
        pos = data.pos
        face = data.face
        edge_attr = data.weight

        vertex2face = self.process_vertex2face(data.vertex2face, data.ptr, data.face_len, data.vertex2face_len)

        pos = self.pos_normalize(pos, batch)
        pos_origin = pos.clone()
        
        if self.mc:
            normal = data.normal
            normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
            normal = normal - normal.mean(dim = 0)
            pos = torch.cat([pos.unsqueeze(-1), normal.unsqueeze(-1)], dim=-1).float()

        area = scatter(self.coord2area(face, pos_origin)[vertex2face[:, 1]], vertex2face[:, 0], dim_size = data.pos.shape[0], reduce='mean' )
        x = self.feat(torch.cat([area.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float())

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = edge_attr, face=face)
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = edge_attr, face=face)
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = edge_attr, face=face)

        x = F.relu(self.ln1(self.lin1(x), batch))


        x = global_mean_pool(x, data.batch)

        x = self.lin2(x)
        return F.log_softmax(x, dim=1) 

class HUMAN(torch.nn.Module):
    def __init__(self, args):
        super(HUMAN, self).__init__()
        
        input_dim = 10
        width = 32
        num_vector = 2

        if 'mc' in args.layer: 
            self.mc = True 
        else:
            self.mc = False
        
        self.hier = args.hier

        self.feat = nn.Linear(input_dim, width)
        
        if args.layer == "EGNN":
            self.conv1 = EGNN(width,   width*2, width,   edges_in_d=1, coords_agg='mean')
            self.conv2 = EGNN(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")  
            self.conv3 = EGNN(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean") 
        elif args.layer == "EGNNmc":
            self.conv1 = EGNNmc(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
            self.conv2 = EGNNmc(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
            self.conv3 = EGNNmc(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  ) 
        elif args.layer == "MeshEGNN":
            self.conv1 = MeshEGNN(width,   width*2, width,   edges_in_d=1, coords_agg='mean')
            self.conv2 = MeshEGNN(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")  
            self.conv3 = MeshEGNN(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean") 
        elif args.layer == "MeshEGNNmc":
            self.conv1 = MeshEGNNmc(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
            self.conv2 = MeshEGNNmc(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
            self.conv3 = MeshEGNNmc(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  )  

        self.pool1 = Pooling(0.25, 0.2, width*8,  width*16)
        self.pool2 = Pooling(0.25, 0.2, width*16, width*32)
        self.pool3 = Pooling(0.25, 0.2, width*32, width*32)
        self.unpool4 = Unpooling(3, width*64, width*16)
        self.unpool5 = Unpooling(3, width*32, width*16)
        self.unpool6 = Unpooling(3, width*24, width*8)

        self.lin1 = nn.Linear(width*8, width*8)
        self.ln1 = LayerNorm(width*8)
        self.lin2 = nn.Linear(width*8, args.target_dim)

    def coord2area(self, face, coord):
        vec1 = coord[face[1]] - coord[face[0]]
        vec2 = coord[face[2]] - coord[face[0]]
        face_norm = vec1.cross(vec2)
        face_area = torch.norm(face_norm, p = 2, dim = -1) / 2 
        return face_area

    def pos_normalize(self, pos, batch):
        centroid = aggregate_mean(pos, batch, batch.max()+1)[batch]
        pos = pos - centroid
        m = aggregate_max(torch.sqrt(torch.sum(pos**2, axis=1)), batch, batch.max()+1)[batch]
        pos = pos / m.unsqueeze(-1)
        return pos

    def process_vertex2face(self, vertex2face, index_vertex, index_face, batch):
        index_batch = torch.arange(len(batch))
        batch = torch.repeat_interleave(index_batch.type_as(batch), batch, dim=0)
        vertex = vertex2face[..., 0]
        face = vertex2face[..., 1]
        vertex += index_vertex[:-1][batch]
        vertex = vertex.flatten()
        face += torch.cumsum(torch.cat([torch.zeros(1).type_as(index_face), index_face]), dim=0)[:-1][batch]
        face = face.flatten()
        return torch.stack([vertex, face]).T

    def forward(self, data):
        batch = data.batch
        pos = data.pos        
        normal = data.normal
        face = data.face
        vertex2face = self.process_vertex2face(data.vertex2face, data.ptr, data.face_len, data.vertex2face_len)

        pos = self.pos_normalize(pos, batch)
        pos_origin = pos.clone()

        if self.mc:
            normal = data.normal
            normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
            normal = normal - normal.mean(dim = 0)
            pos = torch.cat([pos.unsqueeze(-1), normal.unsqueeze(-1)], dim=-1).float()

        area = scatter(self.coord2area(face, pos_origin)[vertex2face[:, 1]], vertex2face[:, 0], dim_size = data.pos.shape[0], reduce='mean' )
        x = self.feat(torch.cat([area.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float())

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long())
        
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long())
        
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long())
    
        x_0, pos_0, pos_origin_0, batch_0 = x, pos, pos_origin, data.batch

        if self.hier:
            pos_origin_1, pos_1, x_1, batch_1 = self.pool1(pos_origin_0, pos_0, x_0, batch_0)
            pos_origin_2, pos_2, x_2, batch_2 = self.pool2(pos_origin_1, pos_1, x_1, batch_1)
            pos_origin_3, pos_3, x_3, batch_3 = self.pool3(pos_origin_2, pos_2, x_2, batch_2)
            x_4                               = self.unpool4(pos_origin_3, x_3, batch_3, pos_origin_2, x_2, batch_2)
            x_5                               = self.unpool5(pos_origin_2, x_4, batch_2, pos_origin_1, x_1, batch_1)
            x_6                               = self.unpool6(pos_origin_1, x_5, batch_1, pos_origin_0, x_0, batch_0)

            x = x_6


        x = aggregate_mean(x[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])
        batch = aggregate_mean(data.batch[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])

        x = self.lin1(x)
        x = F.relu(self.ln1(x, batch))
        x = self.lin2(x)

        return F.log_softmax(x, dim=1) 
    

