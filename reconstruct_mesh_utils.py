import numpy as np
import torch
import open3d as o3d
from utils import umeyama
from tqdm import trange, tqdm
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from pytorch3d.ops import knn_points

device = "cuda"
    

from largesteps.parameterize import from_differential,to_differential
from largesteps.geometry import compute_matrix
from largesteps.geometry import laplacian_uniform
from largesteps.optimize import AdamUniform

def reconstruct_mesh_largesteps_certain_template(points, masks, template_v, template_f, weight_points=10000, weight_laplacian=3000, epochs=1000, return_loss_record = False):
    # prepare some constant
    vertices = template_v
    faces = template_f
    M = compute_matrix(torch.from_numpy(vertices).to(device),torch.from_numpy(faces).to(device),19)
    L = laplacian_uniform(torch.from_numpy(vertices).to(device), torch.from_numpy(faces).to(device))
    c, R, t = umeyama(vertices[masks], points[masks])
    transform_template_mesh_vertices = c * vertices @ R + t
    u = to_differential(M, torch.from_numpy(transform_template_mesh_vertices).to(device).float())
    u.requires_grad_()
    optimizer = AdamUniform([u], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-3)
    target_lmk = torch.from_numpy(points[masks]).to(device).float()
    masks_torch = torch.from_numpy(masks).to(device).bool()
    loss_record = torch.zeros(epochs, device=device)
    for i in (tbar:=trange(epochs)):
        v = from_differential(M, u)
        predict_lmk = v[masks_torch]
        
        
        
        loss_v = torch.nn.functional.mse_loss(predict_lmk, target_lmk, reduction='mean') * weight_points
        loss_laplacian = (L @ v).square().mean() * weight_laplacian
        loss = loss_v + loss_laplacian
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_record[i] = loss.item()
    if return_loss_record:
        return v.detach().cpu().numpy(), loss_record.detach().cpu().numpy()
    else:
        return v.detach().cpu().numpy()