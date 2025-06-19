import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt


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

def shape_loss(source_vertices, target_vertices):
    """
    Compute shape loss between source and target vertices.
    This loss is calculated in the space of target vertices.
    Args:
        source_vertices (torch.Tensor): Source vertices of shape (n, dim).
        target_vertices (torch.Tensor): Target vertices of shape (n, dim).
    Returns:
        loss (torch.Tensor): Computed shape loss.
    """
    # per-vertex loss
    assert source_vertices.shape[0] == target_vertices.shape[0]
    assert source_vertices.shape[1] == target_vertices.shape[1]
    
    # first estimate the scale, rotation and translation
    c, R, t = umeyama(source_vertices.detach().cpu().numpy(), target_vertices.detach().cpu().numpy())
    R = torch.from_numpy(R).to(source_vertices.device)
    t = torch.from_numpy(t).to(source_vertices.device)
    # apply the transformation
    source_vertices = c * source_vertices @ R + t
    loss = torch.mean(torch.sum((source_vertices - target_vertices) ** 2, dim=1))
    return loss

def vertice_loss(source_vertices, target_vertices):
    """
    Compute the loss between two sets of vertices.
    Args:
        source_vertices (torch.Tensor): Source vertices of shape (n, dim).
        target_vertices (torch.Tensor): Target vertices of shape (n, dim).
    Returns:
        loss (torch.Tensor): Computed loss.
    """
    assert source_vertices.shape == target_vertices.shape
    loss = torch.mean(torch.sum((source_vertices - target_vertices) ** 2, dim=1))
    return loss


class SummaryWriter():
    """
    这是我自己做实验的时候不想用tensorboard的一个简单的可视化工具。
    建议使用tensorboard，这个太简单了。
    """
    def __init__(self) -> None:
        self.losses = {}
    
    def add_scalar(self,name,value):
        if name not in self.losses:
            self.losses[name] = []
        self.losses[name].append(value)
    
    def plot(self, start_step=0):
        for key in self.losses.keys():
            plt.plot(self.losses[key][start_step:],label=key)
        plt.legend()
        plt.show()
    


from matplotlib import cm
from matplotlib import pyplot as plt
def apply_color_map(data,cmap = 'viridis', vmin=None, vmax=None):
    """
    将数据根据大小映射成颜色。
    Args:
        data (array-like): 输入数据，1D数组。
        cmap (str): 使用的颜色映射名称，默认为'viridis'。
        vmin (float, optional): 颜色映射的最小值，默认为数据的最小值。
        vmax (float, optional): 颜色映射的最大值，默认为数据的最大值。
    Returns:
        np.ndarray: 应用颜色映射后的RGB颜色数组，形状为 (n, 3)，其中 n 是数据的长度。
    """
    data = np.array(data)
    if vmin is None:
        v_min = np.min(data)
    if vmax is None:
        v_max = np.max(data)
    return getattr(cm,cmap)(plt.Normalize(vmin,vmax)(data))[...,:3]