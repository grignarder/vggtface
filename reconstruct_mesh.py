import numpy as np
import torch
import open3d as o3d
from utils import umeyama
from tqdm import trange, tqdm
from sklearn.neighbors import NearestNeighbors

device = "cuda"

def reconstruct_mesh_Knn(points, uvs, uvs_array, k=1):
    assert points.shape[0] == uvs.shape[0], "points and uvs must have the same number of points"
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(uvs)
    distances, indices = nbrs.kneighbors(uvs_array)
    points_for_reconstruction = points[indices]
    weights_for_reconstruction = 1 / (distances+1e-6)
    # normalize weights
    weights_for_reconstruction /= weights_for_reconstruction.sum(axis=1, keepdims=True)
    vertices_result = (weights_for_reconstruction[...,None] * points_for_reconstruction).sum(axis=1)
    return vertices_result

from largesteps.parameterize import from_differential,to_differential
from largesteps.geometry import compute_matrix
from largesteps.geometry import laplacian_uniform
from largesteps.optimize import AdamUniform

# prepare some constant
template_mesh = o3d.io.read_triangle_mesh("/root/autodl-tmp/facescape/template/ict_template_cut.ply")
vertices = np.asarray(template_mesh.vertices)
faces = np.asarray(template_mesh.triangles)
M = compute_matrix(torch.from_numpy(vertices).to(device),torch.from_numpy(faces).to(device),19)
tempalte_vertices = np.asarray(template_mesh.vertices)
kpt_indices = [1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4734, 4706, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5202, 5189, 2081, 0, 4275, 5694, 5707, 5840, 5955, 5012, 5451, 5335, 5196, 5205, 5027, 5710, 5701, 5964, 5011, 5460]
select_indices = kpt_indices[17:]
select_indices = list(range(6200))
L = laplacian_uniform(torch.from_numpy(vertices).to(device), torch.from_numpy(faces).to(device))

def remove_outliers(points, uvs, remove_percent = 1):
    n_nbrs = int(points.shape[0] * 0.0002)
    nbrs = NearestNeighbors(n_neighbors=n_nbrs, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    distances = distances.mean(axis=1)
    valid_indices = distances < np.percentile(distances, 100 - remove_percent)
    return points[valid_indices], uvs[valid_indices]

def reconstruct_mesh_largesteps(points, uvs, uvs_array, k, return_v_knn=False):
    K = k
    v_knn = reconstruct_mesh_Knn(points, uvs, uvs_array, K)
    c, R, t = umeyama(tempalte_vertices, v_knn)
    transform_template_mesh_vertices = c * tempalte_vertices @ R + t
    u = to_differential(M, torch.from_numpy(transform_template_mesh_vertices).to(device).float())
    u.requires_grad_()
    optimizer = AdamUniform([u], lr=1e-2)
    target_lmk = torch.from_numpy(v_knn[select_indices]).to(device).float()
    for i in (tbar:=trange(1000)):
        v = from_differential(M, u)
        predict_lmk = v[select_indices]
        
        
        
        loss_v = torch.nn.functional.mse_loss(predict_lmk, target_lmk, reduction='mean') * 10000
        loss_laplacian = (L @ v).square().mean() * 3000
        loss = loss_v + loss_laplacian
        
        tbar.set_description(f"loss_v: {loss_v.item():.4f}, loss_laplacian: {loss_laplacian.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if return_v_knn:
        return v.detach().cpu().numpy(), v_knn
    else:
        return v.detach().cpu().numpy()


if __name__ == "__main__":
    import pickle
    import open3d as o3d
    import trimesh
    from trimesh.proximity import closest_point
    from trimesh.triangles import points_to_barycentric

    uvs = pickle.load(open("/root/autodl-tmp/facescape/template/ict_template_cut_uvs.pkl", "rb"))
    uvs_array = np.zeros((uvs.keys().__len__(), 2), dtype=np.float32)
    for i in range(uvs.keys().__len__()):
        uvs_array[i, 0] = uvs[i][0]
        uvs_array[i, 1] = uvs[i][1]
    # load target mesh
    target_mesh = o3d.io.read_triangle_mesh("/root/autodl-tmp/facescape/mesh/2_ictcut.ply")
    # target_mesh = o3d.io.read_triangle_mesh("/root/autodl-tmp/facescape/result/exp1/1_neutral.ply")
    faces = np.asarray(target_mesh.triangles)
    vertices = np.asarray(target_mesh.vertices)
    # sample 30000 points from target mesh
    points = target_mesh.sample_points_uniformly(number_of_points=300000)
    points = np.asarray(points.points)
    # get uvs_gt
    target_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    closest_points, distances, face_indices = closest_point(target_trimesh, points)
    # get triangles according to face_indices
    triangles = target_trimesh.faces[face_indices]
    triangles = target_trimesh.vertices[triangles]
    closet_points_barycentric = points_to_barycentric(triangles, closest_points)
    # get uvs according to barycentric coordinates
    triangles = target_trimesh.faces[face_indices]
    uvs_gt = (closet_points_barycentric[..., None] * uvs_array[triangles]).sum(1)
    point_uvs = uvs_gt
    
    K = 10
    vertices_result = reconstruct_mesh_Knn(points, point_uvs, uvs_array, 1)
    print(vertices_result.shape)
    
    points, point_uvs = remove_outliers(points, point_uvs, remove_percent=1)
    
    vertices_result = reconstruct_mesh_largesteps(points, point_uvs, uvs_array, K, return_v_knn=False)
    print(vertices_result.shape)