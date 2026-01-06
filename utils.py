import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

refine_edges = None
num_vertices_base_resolution = None
num_faces_base_resolution = None


def umeyama(P, Q):
    """
    Estimate a scale, rotation, and translation between two sets of points P and Q which have correspondances.
    Args:
        P (np.ndarray): Source points of shape (n, dim).
        Q (np.ndarray): Target points of shape (n, dim).
    Returns:
        c (float): Scale factor.
        R (np.ndarray): Rotation matrix of shape (dim, dim).
        t (np.ndarray): Translation vector of shape (dim,).
    """
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t


def numpy_to_tensor(numpy_array_dict, device="cuda"):
    """
    将numpy数组转换为torch张量，并将其移动到指定设备。
    Args:
        numpy_array_dict (dict): 包含numpy数组的字典。
        device (str): 目标设备，默认为"cuda"。
    Returns:
        dict: 包含torch张量的字典。
    """
    tensor_dict = {}
    for key, value in numpy_array_dict.items():
        tensor_dict[key] = torch.from_numpy(value).to(device)
    return tensor_dict

def tensor_to_numpy(tensor_dict):
    """
    将torch张量转换为numpy数组。
    Args:
        tensor_dict (dict): 包含torch张量的字典。
    Returns:
        dict: 包含numpy数组的字典。
    """
    numpy_dict = {}
    for key, value in tensor_dict.items():
        numpy_dict[key] = value.detach().cpu().numpy()
    return numpy_dict


from matplotlib import cm
from matplotlib import pyplot as plt
def apply_color_map(data,cmap = 'viridis', vmin=None, vmax=None):
    data = np.array(data)
    if vmin is None:
        v_min = np.min(data)
    if vmax is None:
        v_max = np.max(data)
    return getattr(cm,cmap)(plt.Normalize(vmin,vmax)(data))[...,:3]


def subdivide_meshes(v,f, albedo=None):
    """
    v: (N,3)
    f: (N_faces,3)
    albedo: (N,3)

                   v0
                   /\
                  /  \
                 / f0 \
             v4 /______\ v3
               /\      /\
              /  \ f3 /  \
             / f2 \  / f1 \
            /______\/______\
           v2       v5       v1
    """
    global num_faces_base_resolution, num_vertices_base_resolution
    if num_vertices_base_resolution is None:
        num_vertices_base_resolution = v.shape[0]
    if num_faces_base_resolution is None:
        num_faces_base_resolution = f.shape[0]
    with torch.no_grad():
        v0, v1, v2 = f.chunk(3,dim=1)
        e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
        e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
        e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)
        edges = torch.cat([e12, e20, e01], dim=0)
        edges, _ = edges.sort(dim=1)
        # V = v.shape[0]
        # edges_hash = V * edges[:, 0] + edges[:, 1]
        # u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
        
        # sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
        # unique_mask = torch.ones(
        #     edges_hash.shape[0], dtype=torch.bool, device=v.device
        # )
        # unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
        # unique_idx = sort_idx[unique_mask]

        # edges = torch.stack([u // V, u % V], dim=1)
        
        edges = edges.long()
        E = edges.shape[0]
        V = v.shape[0]
        edges_hash = E * edges[:, 0] + edges[:, 1]
        u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
        sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
        unique_mask = torch.ones(
            edges_hash.shape[0], dtype=torch.bool, device=v.device
        )
        unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
        unique_idx = sort_idx[unique_mask]
        edges = torch.stack([u // E, u % E], dim=1)
        
    
        faces_packed_to_edges = inverse_idxs.reshape(3, f.shape[0]).t() + V
        f0 = torch.stack(
                [
                    f[:, 0],
                    faces_packed_to_edges[:, 2],
                    faces_packed_to_edges[:, 1],
                ],
                dim=1,
            )
        f1 = torch.stack(
                [
                    f[:, 1],
                    faces_packed_to_edges[:, 0],
                    faces_packed_to_edges[:, 2],
                ],
                dim=1,
            )
        f2 = torch.stack(
                [
                    f[:, 2],
                    faces_packed_to_edges[:, 1],
                    faces_packed_to_edges[:, 0],
                ],
                dim=1,
            )
        f3 = faces_packed_to_edges
        subdivided_faces_packed = torch.cat(
                [f0, f1, f2, f3], dim=0
        )
    new_verts = v[edges.long()].mean(dim=1)
    global refine_edges
    if refine_edges is None:
        refine_edges = edges.long().detach().cpu().numpy()
    if albedo == None:
        return torch.cat((v,new_verts),dim=0),subdivided_faces_packed.int(),None
    else:
        new_albedo = albedo[edges.long()].mean(dim=1)
        return torch.cat((v,new_verts),dim=0),subdivided_faces_packed.int(),torch.cat((albedo,new_albedo),dim=0)


def mesh_refiner(v,f,albedo=None,iterations=1,refine_method = subdivide_meshes):
    devided_mesh_v, devided_mesh_f,albedo = refine_method(v,f, albedo=albedo)
    for i in range(iterations-1):
        devided_mesh_v, devided_mesh_f,albedo = refine_method(devided_mesh_v,devided_mesh_f,albedo)
    if albedo == None:
        return devided_mesh_v, devided_mesh_f
    else:
        return devided_mesh_v, devided_mesh_f, albedo
    
    
def quick_refine(features):
    global refine_edges
    if refine_edges is not None:
        try:
            new_features = features[refine_edges].mean(axis=1)
            new_features = np.concatenate((features,new_features),axis=0)
            return new_features
        except:
            # features is torch tensor
            new_features = features[refine_edges].mean(dim=1)
            new_features = torch.cat((features,new_features),dim=0)
            return new_features
    return None