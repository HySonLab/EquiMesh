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
from egnn.nn.transformer_conv import *
from .aggregator import *
from .utils import *

class EGNN(torch.nn.Module):
    def __init__(self, args):
        super(EGNN, self).__init__()
        width = 16

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin
        
        self.feat = nn.Linear(1, width)

        self.conv1 = EGNN_Conv(1, width, width, edges_in_d=1, coords_agg='mean')
        self.conv2 = EGNN_Conv(width, width, width, edges_in_d=1, coords_agg='mean')
        self.conv3 = EGNN_Conv(width, width, width, edges_in_d=1, coords_agg="mean")

        # Dense final layers
        self.lin1 = nn.Linear(width, args.hid_dim)
        self.lin2 = nn.Linear(args.hid_dim, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling

    def compute_face_area(self, data):
        pos, face = data.pos, data.face 
        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]
        face_norm = vec1.cross(vec2)
        face_area = torch.norm(face_norm, p = 2, dim = -1) / 2 
        face_area = face_area.repeat(3)
        idx = face.view(-1)
        area = scatter(face_area, idx, 0,None, pos.size(0), reduce = "sum") 
        return area.unsqueeze(-1)

    def forward(self, data):
        # takes positions as input features
        pos = data.pos
        x = data.area_point.unsqueeze(-1).to(torch.float32)
        
        x, pos = self.conv1(x, edge_index = data.edge_index, coord=pos, edge_attr = data.weight)
        x, pos = self.conv2(x, edge_index = data.edge_index, coord=pos, edge_attr = data.weight)
        x, pos = self.conv3(x, edge_index = data.edge_index, coord=pos, edge_attr = data.weight)

        # x = torch.cat([x, pos], dim = -1)
        x = x

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        if self.do_mean_pooling:
            x = torch.mean(x, dim=0, keepdim=True)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class EGNNMesh(torch.nn.Module):
    def __init__(self, args):
        super(EGNNMesh, self).__init__()
        width = 16

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        self.transform = GemPrecomp(args.n_rings, args.max_order)
        
        self.feat = nn.Linear(1, width)
        #self.feat = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width))
        #self.conv1 = EGNN_Conv(width, width, width, edges_in_d=1)
        self.conv1 = EGNN_Conv(1, width, width, edges_in_d=1, coords_agg='mean')
        self.conv2 = EGNN_Conv(width, width, width, edges_in_d=1, coords_agg='mean')
        self.conv3 = EGNN_Conv(width, width, width, edges_in_d=1, coords_agg="mean")



        # Dense final layers
        self.lin1 = nn.Linear(width, args.hid_dim)
        self.lin2 = nn.Linear(args.hid_dim, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling

    def compute_face_position(self, data):
        pos, face = data.pos, data.face 
        face_pos = pos[face.reshape(-1)].reshape(-1, 3, 3).mean(1)
        data.face_pos = face_pos
        return data

    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        # data0 = self.transform(data)
        data = self.compute_face_position(data)
         
        # attr0 = (data0.edge_index, data0.precomp, data0.precomp_self, data0.connection)

        # takes positions as input features
        pos = data.face_pos

        self.compute_face_position(data)
        x = data.area_face.unsqueeze(-1).to(torch.float32)
        
        x, pos = self.conv1(x, edge_index = data.area_adj.T.long(), coord=pos, edge_attr = torch.ones(data.area_adj.shape[0]).type_as(data.pos))
        x, pos = self.conv2(x, edge_index = data.area_adj.T.long(), coord=pos, edge_attr = torch.ones(data.area_adj.shape[0]).type_as(data.pos))
        x, pos = self.conv3(x, edge_index = data.area_adj.T.long(), coord=pos, edge_attr = torch.ones(data.area_adj.shape[0]).type_as(data.pos))

        # x = torch.cat([x, pos], dim = -1)
        x = torch.matmul(data.vertex2face, x)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        if self.do_mean_pooling:
            x = torch.mean(x, dim=0, keepdim=True)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    
class EGNNAreaPlusS(torch.nn.Module):
    def __init__(self, args):
        super(EGNNAreaPlusS, self).__init__()
        width = 32 # shred

        self.feat = nn.Linear(10, width)
        self.conv1 = EGNN_ConvAreaPlus(width,   width*2, width,   edges_in_d=1, coords_agg='mean')
        self.conv2 = EGNN_ConvAreaPlus(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")  
        self.conv3 = EGNN_ConvAreaPlus(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean") 

        self.lin1 = nn.Linear(width*8, width*8)
        self.lin2 = nn.Linear(width*8, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling
        self.do_face_pooling = hasattr(args, "face_pooling") and args.face_pooling

    
    def coord2area(self, face, coord):
        vec1 = coord[face[1]] - coord[face[0]]
        vec2 = coord[face[2]] - coord[face[0]]
        face_norm = vec1.cross(vec2)
        face_area = torch.norm(face_norm, p = 2, dim = -1) / 2 
        return face_area

    def pos_normalize(self, pos):
        centroid = torch.mean(pos, axis=0)
        pos = pos - centroid
        m = torch.max(torch.sqrt(torch.sum(pos**2, axis=1)))
        pos = pos / m
        return pos

    def forward(self, data):
        # takes positions as input features
        pos = data.pos
        pos = self.pos_normalize(pos)
        area = scatter(self.coord2area(data.face, pos)[data.vertex2face[:, 1]], data.vertex2face[:, 0], dim_size = data.pos.shape[0], reduce='mean' )
        x = self.feat(torch.cat([area.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float())

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)


        x = F.relu(self.lin1(x))
        # x = F.dropout(x, training=self.training)
        if self.do_mean_pooling:
            x = global_mean_pool(x, data.batch)

        if self.do_face_pooling:
            x = aggregate_mean(x[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])

        x = self.lin2(x)

        return F.log_softmax(x, dim=1)   

class EGNNAreaPlusPlus(torch.nn.Module):
    def __init__(self, args):
        super(EGNNAreaPlusPlus, self).__init__()
        width = 16

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        # self.transform = GemPrecomp(args.n_rings, args.max_order)
        
        self.feat = nn.Linear(10, width)
        # self.feat = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width))
        # self.conv1 = EGNN_Conv(width, width, width, edges_in_d=1)
        self.conv1 = EGNN_ConvAreaPlusPlus(10, width, width, edges_in_d=1, coords_agg='mean')
        self.conv2 = EGNN_ConvAreaPlusPlus(width, width, width, edges_in_d=1, coords_agg='mean')
        self.conv3 = EGNN_ConvAreaPlusPlus(width, width, width, edges_in_d=1, coords_agg="mean")
        
        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling
        self.do_face_pooling = hasattr(args, "face_pooling") and args.face_pooling

        # Dense final layers
        # if self.do_face_pooling:
        #     width = width * 4
        self.lin1 = nn.Linear(width, args.hid_dim)
        self.lin2 = nn.Linear(args.hid_dim, args.target_dim)

    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        # data0 = self.transform(data)
        
        # attr0 = (data0.edge_index, data0.precomp, data0.precomp_self, data0.connection)
        # takes positions as input features
        pos = data.pos
        normal = data.vertex_normal
        #x = torch.cat([data.pos, self.feat(data.area_point.unsqueeze(-1).to(torch.float32))], dim=-1)
        #x = self.feat(data.area_point.unsqueeze(-1).to(torch.float32))
        x = torch.cat([data.area_point.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float()
        x_res = self.feat(x)

        pos = pos - pos.mean(dim = 0)
        normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
        normal = normal - normal.mean(dim = 0)
        x, pos, normal = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, normal=normal, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        pos = pos - pos.mean(dim = 0)
        normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
        normal = normal - normal.mean(dim = 0)
        x, pos, normal = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, normal=normal, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        pos = pos - pos.mean(dim = 0)
        normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
        normal = normal - normal.mean(dim = 0)
        x, pos, normal = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, normal=normal, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)

        if self.do_mean_pooling:
            x = global_mean_pool(x, data.batch)
        if self.do_face_pooling:
            import pdb; pdb.set_trace()
            x = x + x_res
            x = aggregate_mean(x[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])
            
        x = F.silu(self.lin1(x))
        # x = F.dropout(x, training=self.training) 
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)   

class EGNNMC(torch.nn.Module):
    def __init__(self, args):
        super(EGNNMC, self).__init__()
        width = 16

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        self.transform = GemPrecomp(args.n_rings, args.max_order)
        
        self.feat = nn.Linear(1, width)

        num_vector = 2
        self.conv1 = EGNN_Conv_MC(1, width, width, edges_in_d=1, coords_agg='mean', num_vectors_in=1, num_vectors_out=num_vector)
        self.conv2 = EGNN_Conv_MC(width, width, width, edges_in_d=1, coords_agg='mean', num_vectors_in=num_vector, num_vectors_out=num_vector)
        self.conv3 = EGNN_Conv_MC(width, width, width, edges_in_d=1, coords_agg="mean", last_layer=True, num_vectors_in=num_vector, num_vectors_out=1)



        # Dense final layers
        self.lin1 = nn.Linear(width, args.hid_dim)
        self.lin2 = nn.Linear(args.hid_dim, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling

    def compute_face_area(self, data):
        pos, face = data.pos, data.face 
        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]
        face_norm = vec1.cross(vec2)
        face_area = torch.norm(face_norm, p = 2, dim = -1) / 2 
        face_area = face_area.repeat(3)
        idx = face.view(-1)
        area = scatter(face_area, idx, 0,None, pos.size(0), reduce = "sum") 
        return area.unsqueeze(-1)

    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        data0 = self.transform(data)
         
        attr0 = (data0.edge_index, data0.precomp, data0.precomp_self, data0.connection)

        # takes positions as input features
        pos = data.pos
        #x = torch.cat([data.pos, self.feat(data.area_point.unsqueeze(-1).to(torch.float32))], dim=-1)
        #x = self.feat(data.area_point.unsqueeze(-1).to(torch.float32))
        x = data.area_point.unsqueeze(-1).to(torch.float32)
        
        x, pos = self.conv1(x, edge_index = data.edge_index, coord=pos, edge_attr = data.weight)
        x, pos = self.conv2(x, edge_index = data.edge_index, coord=pos, edge_attr = data.weight)
        x, pos = self.conv3(x, edge_index = data.edge_index, coord=pos, edge_attr = data.weight)

        # x = torch.cat([x, pos], dim = -1)
        x = x

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        if self.do_mean_pooling:
            x = global_mean_pool(x, data.batch)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
  
class EGNNAreaPlusMC(torch.nn.Module):
    def __init__(self, args):
        super(EGNNAreaPlusMC, self).__init__()
        # width = 16 # tosca faust
        width = 32 # shred

        num_vector = 2
        self.feat = nn.Linear(10, width)
        self.conv1 = EGNN_ConvAreaPlus_MC(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
        self.conv2 = EGNN_ConvAreaPlus_MC(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
        self.conv3 = EGNN_ConvAreaPlus_MC(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  ) 

        self.pool1 = Pooling(0.25, 0.2, width*8,  width*16)
        self.pool2 = Pooling(0.25, 0.2, width*16, width*32)

        self.lin1 = nn.Linear(width*8, width*8)
        self.lin2 = nn.Linear(width*8, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling
        self.do_face_pooling = hasattr(args, "face_pooling") and args.face_pooling

    def forward(self, data):
        # takes positions as input features
        pos = data.pos
        normal = data.vertex_normal
        x = self.feat(torch.cat([data.area_point.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float())

        pos = pos - pos.mean(dim = 0)
        normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
        normal = normal - normal.mean(dim = 0)
        pos = torch.cat([pos.unsqueeze(-1), normal.unsqueeze(-1)], dim=-1).float()

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)

        x_0, pos_0, pos_origin_0, batch_0 = x, pos, data.pos, data.batch

        # pos_origin_1, pos_1, x_1, batch_1 = self.pool1(pos_origin_0, pos_0, x_0, batch_0)
        # pos_origin_2, pos_2, x_2, batch_2 = self.pool2(pos_origin_1, pos_1, x_1, batch_1)

        # x, batch = x_2, batch_2

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, training=self.training)
        if self.do_mean_pooling:
            x = global_mean_pool(x, batch_0)

        if self.do_face_pooling:
            x = aggregate_mean(x[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])

        x = self.lin2(x)
        return F.log_softmax(x, dim=1) 

class EGNNAreaPlusMCS(torch.nn.Module):
    def __init__(self, args):
        super(EGNNAreaPlusMCS, self).__init__()
        self.hks = args.hks

        input_dim = 10 if self.hks else 1     
        width = args.hidden_dim

        num_vector = 2
        self.feat = nn.Linear(input_dim, width)
        self.conv1 = EGNN_ConvAreaPlus_MC(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
        self.conv2 = EGNN_ConvAreaPlus_MC(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
        self.conv3 = EGNN_ConvAreaPlus_MC(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  ) 

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

    def pos_normalize(self, pos):
        centroid = torch.mean(pos, axis=0)
        pos = pos - centroid
        m = torch.max(torch.sqrt(torch.sum(pos**2, axis=1)))
        pos = pos / m
        return pos

    def forward(self, data):
        # takes positions as input features
        pos = data.pos
        face = data.face.long()
        normal = data.normal

        pos = self.pos_normalize(pos)
        normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
        normal = normal - normal.mean(dim = 0)
        pos_origin = pos.clone()
        pos = torch.cat([pos.unsqueeze(-1), normal.unsqueeze(-1)], dim=-1).float()

        area = scatter(self.coord2area(face, pos_origin)[data.vertex2face[:, 1]], data.vertex2face[:, 0], dim_size = pos.shape[0], reduce='mean' )

        if self.hks:
            x = self.feat(torch.cat([area.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float())
        else:
            x = self.feat(area.unsqueeze(-1))

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)

        x = F.relu(self.ln1(self.lin1(x)))

        if self.do_mean_pooling:
            x = global_mean_pool(x, data.batch)

        x = self.lin2(x)
        return F.log_softmax(x, dim=1) 

class EGNNAreaPlusHuman(torch.nn.Module):
    def __init__(self, args):
        super(EGNNAreaPlusHuman, self).__init__()
        # width = 16 # tosca faust
        # width = 32 # shred
        width = 64

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        # self.transform = GemPrecomp(args.n_rings, args.max_order)
        
        self.feat = nn.Linear(10, width)
        self.conv1 = EGNN_ConvAreaPlus(width, width*2, width, edges_in_d=1, coords_agg='mean')
        self.conv2 = EGNN_ConvAreaPlus(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean")
        self.conv3 = EGNN_ConvAreaPlus(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean")

        # Dense final layers
        self.lin1 = nn.Linear(width*8, width*8)
        self.lin2 = nn.Linear(width*8, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling
        self.do_face_pooling = hasattr(args, "face_pooling") and args.face_pooling

    def forward(self, data):
        pos = data.pos

        x = self.feat(torch.cat([data.area_point.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float())

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)


        if self.do_mean_pooling:
            x = global_mean_pool(x, data.batch)

        if self.do_face_pooling:
            x = aggregate_max(x[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])

        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)  

class EGNNAreaPlusHumanMC(torch.nn.Module):
    def __init__(self, args):
        super(EGNNAreaPlusHumanMC, self).__init__()
        # width = 16 # tosca faust
        # width = 32 # shred
        width = 64

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        # self.transform = GemPrecomp(args.n_rings, args.max_order)
        num_vector = 2
        self.feat = nn.Linear(10, width)
        self.conv1 = EGNN_ConvAreaPlus_MC(width, width*2, width, edges_in_d=1, coords_agg='mean', num_vectors_in=2, num_vectors_out=num_vector)
        self.conv2 = EGNN_ConvAreaPlus_MC(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector, num_vectors_out=num_vector*2)
        self.conv3 = EGNN_ConvAreaPlus_MC(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector*2, num_vectors_out=num_vector)
        self.conv4 = EGNN_ConvAreaPlus_MC(width*8, width*4, width*8, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector, num_vectors_out=1, last_layer=True)

        # Dense final layers
        self.lin1 = nn.Linear(width*4, width*4)
        self.ln1 = LayerNorm(width*4)
        self.lin2 = nn.Linear(width*4, args.target_dim)
        # self.ln2 = LayerNorm(width*4)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling
        self.do_face_pooling = hasattr(args, "face_pooling") and args.face_pooling

    def forward(self, data):
        pos = data.pos
        normal = data.vertex_normal

        x = self.feat(torch.cat([data.area_point.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float())

        pos = pos - pos.mean(dim = 0)
        normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
        normal = normal - normal.mean(dim = 0)
        pos = torch.cat([pos.unsqueeze(-1), normal.unsqueeze(-1)], dim=-1).float()

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        
        pos = pos - pos.mean(dim = 0)
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        
        pos = pos - pos.mean(dim = 0)
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
    
        pos = pos - pos.mean(dim = 0)
        x, pos = self.conv4(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                    vertex2face=data.vertex2face.long(), di_angle=data.di_angles)


        if self.do_mean_pooling:
            x = global_mean_pool(x, data.batch)

        if self.do_face_pooling:
            x = aggregate_mean(x[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])
            batch = aggregate_mean(data.batch[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])
        else:
            batch = data.batch

        x = self.lin1(x)
        x = F.relu(self.ln1(x, batch))
        # x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        # x = self.ln2(x, batch)

        return F.log_softmax(x, dim=1)  

class EGNNAreaPlusHumanMCHier(torch.nn.Module):
    def __init__(self, args):
        super(EGNNAreaPlusHumanMCHier, self).__init__()
        # width = 16 # tosca faust
        # width = 32 # shred
        width = 32

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        # self.transform = GemPrecomp(args.n_rings, args.max_order)
        num_vector = 2
        self.feat = nn.Linear(10, width)
        self.conv1 = EGNN_ConvAreaPlus_MC(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
        self.conv2 = EGNN_ConvAreaPlus_MC(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
        self.conv3 = EGNN_ConvAreaPlus_MC(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  )  

        self.pool1 = Pooling(0.25, 0.2, width*8,  width*16)
        self.pool2 = Pooling(0.25, 0.2, width*16, width*32)
        self.unpool3 = Unpooling(3, width*48, width*16)
        self.unpool4 = Unpooling(3, width*24, width*8)

        # Dense final layers
        self.lin1 = nn.Linear(width*8, width*8)
        self.ln1 = LayerNorm(width*8)
        self.lin2 = nn.Linear(width*8, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling
        self.do_face_pooling = hasattr(args, "face_pooling") and args.face_pooling

    def coord2area(self, face, coord):
        vec1 = coord[face[1]] - coord[face[0]]
        vec2 = coord[face[2]] - coord[face[0]]
        face_norm = vec1.cross(vec2)
        face_area = torch.norm(face_norm, p = 2, dim = -1) / 2 
        return face_area

    def pos_normalize(self, pos):
        centroid = torch.mean(pos, axis=0)
        pos = pos - centroid
        m = torch.max(torch.sqrt(torch.sum(pos**2, axis=1)))
        pos = pos / m
        return pos

    def forward(self, data):
        batch = data.batch
        pos = data.pos        
        normal = data.vertex_normal

        pos = self.pos_normalize(pos)
        normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
        normal = normal - normal.mean(dim = 0)
        pos = torch.cat([pos.unsqueeze(-1), normal.unsqueeze(-1)], dim=-1).float()

        area = scatter(self.coord2area(data.face, data.pos)[data.vertex2face[:, 1]], data.vertex2face[:, 0], dim_size = data.pos.shape[0], reduce='mean' )
        x = self.feat(torch.cat([area.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float())

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        
        # pos = pos - pos.mean(dim = 0)
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        
        # pos = pos - pos.mean(dim = 0)
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
    
        x_0, pos_0, pos_origin_0, batch_0 = x, pos, data.pos, data.batch

        pos_origin_1, pos_1, x_1, batch_1 = self.pool1(pos_origin_0, pos_0, x_0, batch_0)
        pos_origin_2, pos_2, x_2, batch_2 = self.pool2(pos_origin_1, pos_1, x_1, batch_1)
        x_3                               = self.unpool3(pos_origin_2, x_2, batch_2, pos_origin_1, x_1, batch_1)
        x_4                               = self.unpool4(pos_origin_1, x_3, batch_1, pos_origin_0, x_0, batch_0)

        x = x_4

        if self.do_mean_pooling:
            x = global_mean_pool(x, data.batch)

        if self.do_face_pooling:
            x = aggregate_mean(x[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])
            batch = aggregate_mean(data.batch[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])
        else:
            batch = data.batch

        x = self.lin1(x)
        x = F.relu(self.ln1(x, batch))
        # x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        # x = self.ln2(x, batch)

        return F.log_softmax(x, dim=1)  

class EGNNAreaPlusHumanMCHierS(torch.nn.Module):
    def __init__(self, args):
        super(EGNNAreaPlusHumanMCHierS, self).__init__()
        # width = 16 # tosca faust
        # width = 32 # shred
        width = 32

        kwargs = dict(
            n_rings=args.n_rings,
            band_limit=args.max_order,
            num_samples=7,
            checkpoint=True,
            batch=args.model_batch,
            n_heads=args.n_heads,
        )

        if hasattr(args, "equiv_bias"):
            kwargs["equiv_bias"] = args.equiv_bias
        if hasattr(args, "regular_non_lin"):
            kwargs["regular_non_lin"] = args.regular_non_lin

        # self.transform = GemPrecomp(args.n_rings, args.max_order)
        num_vector = 2
        self.feat = nn.Linear(10, width)
        self.conv1 = EGNN_ConvAreaPlus_MC(width,   width*2, width,   edges_in_d=1, coords_agg='mean', num_vectors_in=2,            num_vectors_out=num_vector                    )
        self.conv2 = EGNN_ConvAreaPlus_MC(width*2, width*4, width*2, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=num_vector                    )  
        self.conv3 = EGNN_ConvAreaPlus_MC(width*4, width*8, width*4, edges_in_d=1, coords_agg="mean", num_vectors_in=num_vector,   num_vectors_out=1,          last_layer= True  )  

        self.pool1 = Pooling(0.25, 0.2, width*8,  width*16)
        self.pool2 = Pooling(0.25, 0.2, width*16, width*32)
        self.unpool3 = Unpooling(3, width*48, width*16)
        self.unpool4 = Unpooling(3, width*24, width*8)

        # Dense final layers
        self.lin1 = nn.Linear(width*8, width*8)
        self.ln1 = LayerNorm(width*8)
        self.lin2 = nn.Linear(width*8, args.target_dim)

        self.do_mean_pooling = hasattr(args, "mean_pooling") and args.mean_pooling
        self.do_face_pooling = hasattr(args, "face_pooling") and args.face_pooling

    def coord2area(self, face, coord):
        vec1 = coord[face[1]] - coord[face[0]]
        vec2 = coord[face[2]] - coord[face[0]]
        face_norm = vec1.cross(vec2)
        face_area = torch.norm(face_norm, p = 2, dim = -1) / 2 
        return face_area

    def pos_normalize(self, pos):
        centroid = torch.mean(pos, axis=0)
        pos = pos - centroid
        m = torch.max(torch.sqrt(torch.sum(pos**2, axis=1)))
        pos = pos / m
        return pos

    def forward(self, data):
        batch = data.batch
        pos = data.pos        
        normal = data.vertex_normal

        pos = self.pos_normalize(pos)
        normal = normal / (normal.norm(dim=-1) + 1e-6).unsqueeze(-1)
        normal = normal - normal.mean(dim = 0)
        pos = torch.cat([pos.unsqueeze(-1), normal.unsqueeze(-1)], dim=-1).float()

        area = scatter(self.coord2area(data.face, data.pos)[data.vertex2face[:, 1]], data.vertex2face[:, 0], dim_size = data.pos.shape[0], reduce='mean' )
        x = self.feat(torch.cat([area.unsqueeze(-1).to(torch.float32), data.hks], dim = -1).float())

        x, pos = self.conv1(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        
        # pos = pos - pos.mean(dim = 0)
        x, pos = self.conv2(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
        
        # pos = pos - pos.mean(dim = 0)
        x, pos = self.conv3(x, edge_index = data.edge_index.long(), coord=pos, edge_attr = data.weight, face=data.face.long(), 
                            vertex2face=data.vertex2face.long(), di_angle=data.di_angles)
    

        x_0 = aggregate_mean(x[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])
        pos_0 = aggregate_mean(pos[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])
        batch_0 = aggregate_mean(data.batch[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])
        pos_origin_0 = aggregate_mean(data.pos[data.vertex2face[:, 0]], data.vertex2face[:, 1], dim_size = data.face.shape[1])

        # x_0, pos_0, pos_origin_0, batch_0 = x, pos, data.pos, data.batch

        pos_origin_1, pos_1, x_1, batch_1 = self.pool1(pos_origin_0, pos_0, x_0, batch_0)
        pos_origin_2, pos_2, x_2, batch_2 = self.pool2(pos_origin_1, pos_1, x_1, batch_1)
        x_3                               = self.unpool3(pos_origin_2, x_2, batch_2, pos_origin_1, x_1, batch_1)
        x_4                               = self.unpool4(pos_origin_1, x_3, batch_1, pos_origin_0, x_0, batch_0)

        x = x_4
        batch = batch_0

        x = self.lin1(x)
        x = F.relu(self.ln1(x, batch))
        # x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        # x = self.ln2(x, batch)

        return F.log_softmax(x, dim=1)  
  