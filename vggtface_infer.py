import os
import numpy as np
import torch
import open3d as o3d
from matplotlib import pyplot as plt
import cv2
import random
import sys

sys.path.append("./third_party") # so that we can import from third_party folder

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_adaptive_thre
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import pycolmap
from vggt.dependency.np_to_pycolmap import project_3D_points_np

from reconstruct_mesh_utils import (
    NearestNeighbors,
    reconstruct_mesh_largesteps_certain_template
)

import trimesh
from tqdm import tqdm, trange

import time

from utils import mesh_refiner
import utils
import pickle

import argparse

from load_data_utils import load_data, load_vggt_models


# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

with open("./assets/boundary_indices.pkl", 'rb') as f:
    indices_dict = pickle.load(f)
right_eye_indices = indices_dict['right_eye_indices']
left_eye_indices = indices_dict['left_eye_indices']
mouth_indices = indices_dict['mouth_indices_refine']
mouth_inner_indices = indices_dict['mouth_inner_indices_refine']
mouth_inner_inner_indices = indices_dict['mouth_inner_inner_indices_refine']
mask_mouth_inner_vertices = True
mask_indices = mouth_inner_inner_indices
remove_mouth_inner_vertices = True
remove_indices = mouth_indices

def load_template(template="flame", refine_mesh = True):
    """
    加载template mesh的信息，template可选为"flame","ict". refine表示是否加密mesh。
    
    返回值：
        uvs_array: (N, 2) UV坐标数组
        faces: (M, 3) 面片索引数组
        vertices: (N, 3) 顶点坐标数组
    """
    if template == "flame":
        uvs_array = np.load("./assets/flame_cut_uvs.npy")
        template = o3d.io.read_triangle_mesh("./assets/flame_cut_fill_nose.ply")
        uvs_array[:,0] = 1.0075 - uvs_array[:,0]  # flip x-axis to match the template

        faces = np.array(template.triangles)
        vertices = np.array(template.vertices)
    elif template == "ict":
        raise NotImplementedError("ICT template loading is not implemented yet.")
    # if template is a string and end with ".ply"
    elif template.endswith(".ply"):
        # check if the file exists
        if not os.path.exists(template):
            raise FileNotFoundError(f"Template file {template} does not exist.")
        uvs_array = np.load(template.replace(".ply", "_uvs.npy"))
        template = o3d.io.read_triangle_mesh(template)
        faces = np.array(template.triangles)
        vertices = np.array(template.vertices)
    else:
        raise ValueError("Unsupported template type. Choose 'flame' or 'ict'.")
    if refine_mesh:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        v = np.array(template.vertices)
        f = np.array(template.triangles)
        refined_v, refined_f, refined_uv = mesh_refiner(torch.from_numpy(v).to(device),
                                                        torch.from_numpy(f).to(device),
                                                        torch.from_numpy(uvs_array).to(device),iterations=1)
        faces = refined_f.cpu().numpy()
        uvs_array = refined_uv.cpu().numpy()
        vertices = refined_v.cpu().numpy()
    return uvs_array, faces, vertices


def vggt_infer(model, data):
    imgs = data["imgs"]  # (N, H, W, C)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            images = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2)  # (B, C, H, W)
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images.float())
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        # # Predict Point Maps
        # point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
            
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_maps = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))
    
    vggt_outputs = {}
    vggt_outputs["depth_map"] = depth_map
    vggt_outputs["depth_conf"] = depth_conf
    vggt_outputs["point_maps"] = point_maps
    vggt_outputs["extrinsic"] = extrinsic
    vggt_outputs["intrinsic"] = intrinsic
    
    return vggt_outputs




def generate_pixel_coordinates(h, w):
    x_coords = np.linspace(0.5, w - 0.5, w)
    y_coords = np.linspace(0.5, h - 0.5, h)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    
    coordinates = np.stack([X, Y], axis=-1)  # 注意这里是Y在前，X在后，以匹配[h, w]的顺序
    
    return coordinates

def extract_extri_intri_from_reconstruction(reconstruction):
    extrinsic_list = []
    intrinsic_list = []
    for image_id in sorted(reconstruction.images):
        img = reconstruction.images[image_id]

        # cam_from_world: 3x4
        m34 = img.cam_from_world.matrix()
        m4 = np.eye(4, dtype=np.float64)
        m4[:3, :4] = m34
        extrinsic_list.append(m4[:3, :4])

        cam = reconstruction.cameras[img.camera_id]
        intrinsic_list.append(cam.calibration_matrix())

    return np.stack(extrinsic_list, axis=0), np.stack(intrinsic_list, axis=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='vggtface')
    parser.add_argument('--BASE_PATHS', type=str, default="", help='Base paths for the dataset')
    parser.add_argument('--KNN', type=int, default=5, help='Number of nearest neighbors for reconstruction')
    parser.add_argument('--mx_reproj_error', type=float, default=8.0, help='Maximum reprojection error for bundle adjustment')
    parser.add_argument('--min_inlier_per_frame', type=int, default=256, help='Minimum number of inliers per frame for bundle adjustment')
    parser.add_argument('--n_iters_topBA', type=int, default=1, help="Number of topBA refinement rounds (>=1). ")
    # for n_iters_topBA, we found that 1 is usually enough, 2 can be slightly better
    args = parser.parse_args()
    
    model = load_vggt_models()
    
    uvs_array, template_faces, template_vertices = load_template("./assets/flame_template.ply",refine_mesh=True)
    
    BASE_PATHS = args.BASE_PATHS.split(",")
    KNN = args.KNN
    FACE_MASK_THRE = 0.5
    
    for BASE_PATH in BASE_PATHS:
        try:
            ######################
            # load data
            data = load_data(BASE_PATH, ["imgs", "uvs", "masks"])
            
            # cut data so that it has 16 views
            # you can remove it if you want to use all views
            if data['imgs'].shape[0] > 16:
                data["imgs"] = data["imgs"][:16]
                data["uvs"] = data["uvs"][:16]
                data["masks"] = data["masks"][:16]
        

            ######################
            # VGGT infer
            vggt_outputs = vggt_infer(model, data)
            extrinsic = vggt_outputs["extrinsic"][0]
            intrinsic = vggt_outputs["intrinsic"][0]
            intrinsic = intrinsic.cpu().numpy()
            extrinsic = extrinsic.cpu().numpy()

            ######################
            # make tracks
            n_views, h, w, _ = data["imgs"].shape
            track_coord = generate_pixel_coordinates(h, w)
            track_coord = track_coord[None, ...]  # [1,h,w,2]

            ######################
            # data for BA
            pred_tracks = []  # [nview,v,2]
            pred_vis_scores = []  # [nview,v]
            points_3d = []  # [nview,v,3]

            for i in tqdm(range(n_views)):
                imgs_single_view = data["imgs"][i:i+1]
                masks_single_view = data["masks"][i:i+1]
                point_maps_single_view = vggt_outputs["point_maps"][i:i+1]
                uvs_single_view = data["uvs"][i:i+1]
                
                valid_points = point_maps_single_view[masks_single_view > FACE_MASK_THRE]
                valid_uvs = uvs_single_view[masks_single_view > FACE_MASK_THRE]
                valid_tracks = track_coord[masks_single_view > FACE_MASK_THRE]
                
                nbrs = NearestNeighbors(n_neighbors=KNN, algorithm='auto').fit(valid_uvs)
                distances, indices = nbrs.kneighbors(uvs_array)
                points_for_reconstruction = valid_points[indices]
                weights_for_reconstruction = 1 / (distances + 1e-6)
                tracks_for_reconstruction = valid_tracks[indices]
                
                # normalize weights
                weights_for_reconstruction /= weights_for_reconstruction.sum(axis=1, keepdims=True)
                vertices_result = (weights_for_reconstruction[...,None] * points_for_reconstruction).sum(axis=1)  # [v,3]
                tracks_results = (weights_for_reconstruction[...,None] * tracks_for_reconstruction).sum(axis=1)  # [v,2]
                
                distances_max = distances.max(axis=1)
                vertices_mask = distances_max < np.percentile(distances_max, 70)  # [v]  # visibility in the current view
                
                points_3d.append(vertices_result)
                pred_vis_scores.append(vertices_mask * 1.0)
                pred_tracks.append(tracks_results)


            points_3d = np.stack(points_3d, axis=0)  # [n,v,3]
            pred_vis_scores = np.stack(pred_vis_scores, axis=0)  # [n,v]
            pred_tracks = np.stack(pred_tracks, axis=0)  # [n,v,2]


            # find the one has minimal projection error
            proj_error_list = []
            for i in range(n_views):
                cur_3d_proj, _ = project_3D_points_np(points_3d[i], extrinsic, intrinsic)  # [n,v,2]
                cur_error = np.sum((cur_3d_proj - pred_tracks) ** 2, axis=-1) ** 0.5  # [n,v]
                cur_error = np.sum(cur_error * pred_vis_scores, axis=0)
                proj_error_list.append(cur_error)
            proj_error_list = np.stack(proj_error_list, axis=0)  # [n,v]
            print(proj_error_list.shape)

            select_vert_list = []
            for i in range(proj_error_list.shape[1]):
                min_error_index = np.argmin(proj_error_list[:, i])
                select_vert = points_3d[min_error_index, i]  # [3]
                select_vert_list.append(select_vert)
            select_vert_list = np.stack(select_vert_list, axis=0)  # [v,

            print(select_vert_list.shape)
            points_3d = select_vert_list  # [v,3]
            # points_3d = np.sum(points_3d * pred_vis_scores[..., None], axis=0) / np.sum(pred_vis_scores[..., None], axis=0)  # [v,3]

            print(points_3d.shape)
            print(pred_vis_scores.shape)
            print(pred_tracks.shape)
            
            
            ba_rounds = max(1, int(args.n_iters_topBA))

            image_size = np.array([518, 518])
            shared_camera = False
            camera_type = "SIMPLE_PINHOLE"
            max_reproj_error = args.mx_reproj_error
            min_inlier_per_frame = args.min_inlier_per_frame

            track_mask = pred_vis_scores > 0.5

            points_seed = points_3d
            extrinsic_cur = extrinsic
            intrinsic_cur = intrinsic

            ba_options = pycolmap.BundleAdjustmentOptions()
            # ba_options.refine_principal_point = True

            for round_idx in range(ba_rounds):
                reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                    points_seed,
                    extrinsic_cur,
                    intrinsic_cur,
                    pred_tracks,
                    image_size,
                    masks=track_mask,
                    max_reproj_error=max_reproj_error,
                    shared_camera=shared_camera,
                    camera_type=camera_type,
                    min_inlier_per_frame=min_inlier_per_frame,
                    points_rgb=None,
                )

                pycolmap.bundle_adjustment(reconstruction, ba_options)

                points_3d_ba = []
                for point3d_id in sorted(reconstruction.points3D):
                    point3d = reconstruction.points3D[point3d_id]
                    points_3d_ba.append(point3d.xyz)
                points_3d_ba = np.array(points_3d_ba)

                points_final_ba = np.zeros_like(points_seed)
                points_final_ba[valid_track_mask] = points_3d_ba

                valid_mask_for_mesh = valid_track_mask.copy()
                if mask_mouth_inner_vertices:
                    valid_mask_for_mesh[mask_indices] = 0

                v_largesteps, loss_record = reconstruct_mesh_largesteps_certain_template(
                    points_final_ba,
                    valid_mask_for_mesh.astype(bool),
                    template_vertices,
                    template_faces,
                    weight_laplacian=3000,
                    epochs=400,
                    return_loss_record=True
                )

                # largesteps mesh
                final_mesh_largesteps = o3d.geometry.TriangleMesh()
                final_mesh_largesteps.vertices = o3d.utility.Vector3dVector(v_largesteps)
                final_mesh_largesteps.triangles = o3d.utility.Vector3iVector(template_faces)

                if remove_mouth_inner_vertices:
                    final_mesh_largesteps.remove_vertices_by_index(remove_indices)
                    final_mesh_largesteps.remove_degenerate_triangles()

                largesteps_mesh_path = os.path.join(BASE_PATH, f"result.ply")
                o3d.io.write_triangle_mesh(largesteps_mesh_path, final_mesh_largesteps)

                if round_idx < ba_rounds - 1:
                    extrinsic_cur, intrinsic_cur = extract_extri_intri_from_reconstruction(reconstruction)
                    points_seed = v_largesteps
            
            
        except Exception as e:
            print(e)
    