import torch
import numpy as np
import trimesh
import torch
from torch_geometric.utils import scatter
from .signature import SignatureExtractor
import fast_simplification

def compute_normals_edges_from_mesh(data, normal = True):
    mesh = trimesh.Trimesh(
        vertices=data.pos.numpy(), faces=data.face.numpy().T, process=False
    )
    data.normal = torch.tensor(
        mesh.vertex_normals.copy(), dtype=data.pos.dtype, device=data.pos.device
    )
    data.edge_index = torch.tensor(
        mesh.edges.T.copy(), dtype=torch.long, device=data.pos.device
    )
    idx_to, idx_from = data.edge_index
    num_neighbours = scatter(torch.ones_like(idx_from), index=idx_to, reduce="sum")
    data.weight = 1.0 / num_neighbours[idx_to]
    data.face_len = data.face.numpy().T.shape[0]
    return data


def compute_area_from_mesh(data):

    mesh = trimesh.Trimesh(
        vertices=data.pos.numpy(), faces=data.face.numpy().T, process=False
    )

    vertex = torch.arange(mesh.vertex_faces.shape[0]).unsqueeze(-1).expand(-1, mesh.vertex_faces.shape[1]).flatten().unsqueeze(-1)
    face = torch.tensor(mesh.vertex_faces.reshape(-1)).unsqueeze(-1)
    vertex2face = torch.cat([vertex, face], dim=-1)
    vertex2face = vertex2face[vertex2face[:, 1] != -1]

    area_face = torch.tensor(mesh.area_faces).float()
    data.area_point = scatter(area_face[vertex2face[:, 1]], vertex2face[:, 0], dim=0, dim_size=mesh.vertices.shape[0]).to(data.pos.device)

    data.vertex2face = vertex2face.to(data.pos.device)
    data.vertex2face_len = len(vertex2face)

    return data

def compute_triangle_info_from_mesh(data):

    mesh = trimesh.Trimesh(
        vertices=data.pos.numpy(), faces=data.face.numpy().T, process=False
    )

    data.area_face = torch.tensor(mesh.area_faces, dtype=data.pos.dtype, device=data.pos.device)
    data.area_adj = torch.tensor(mesh.face_adjacency, dtype=data.pos.dtype, device=data.pos.device)
    return data

def compute_di_angle(data):
    mesh = trimesh.Trimesh(
        vertices=data.pos.numpy(), faces=data.face.numpy().T, process=False
    )
    
    dihedral_angle = list()
    for i in range(mesh.faces.shape[0]):
        dihedral_angle.append(list())

    face_adjacency = mesh.face_adjacency

    for adj_faces in face_adjacency:
        dihedral_angle[adj_faces[0]].append(
            np.abs(np.dot(mesh.face_normals[adj_faces[0]], mesh.face_normals[adj_faces[1]])))
        dihedral_angle[adj_faces[1]].append(
            np.abs(np.dot(mesh.face_normals[adj_faces[0]], mesh.face_normals[adj_faces[1]])))

    for i, l in enumerate(dihedral_angle):
        if (len(l)) == 3:
            continue
        l.append(1)
        if (len(l)) == 3:
            continue
        l.append(1)
        if (len(l)) == 3:
            continue
        l.append(1)
        if (len(l)) != 3:
            print(i, 'Padding Failed')
    face_dihedral_angle = np.array(dihedral_angle).reshape(-1, 3)

    data.di_angles = torch.tensor(face_dihedral_angle)

    return data

def compute_hks(data):
    mesh = trimesh.Trimesh(
        vertices=data.pos.numpy(), faces=data.face.numpy().T, process=False
    )

    extractor = SignatureExtractor(mesh, 100, approx='mesh') 
    hks = torch.tensor(extractor.signatures(21, 'heat', return_x_ticks=False))
    hks_cat = ((hks[:, 1] - hks[:, 1].min()) / (hks[:, 1].max() - hks[:, 1].min())).unsqueeze(1)
    for i, k in enumerate([2, 3, 4, 5, 8, 10, 15, 20]):
        hks_norm = ((hks[:, k] - hks[:, k].min()) / (hks[:, k].max() - hks[:, k].min())).unsqueeze(1)
        hks_cat = torch.cat((hks_cat, hks_norm), dim=1)
    
    data.hks = hks_cat

    return data

def compute_hierarchy(data):
    pos_0, face_0 = data.pos.numpy(), data.face.numpy().T
    pos_1, face_1 = fast_simplification.simplify(pos_0, face_0, 0.75)
    pos_2, face_2 = fast_simplification.simplify(pos_1, face_1, 0.75)
    pos_3, face_3 = fast_simplification.simplify(pos_2, face_2, 0.75)
    
    data.hierarchy = [[pos_1, face_1], [pos_2, face_2], [pos_3, face_3]]

    return data