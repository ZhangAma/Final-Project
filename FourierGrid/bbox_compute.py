import torch
import numpy as np
from FourierGrid import utils, dvgo
import time
from FourierGrid.load_everything import load_existing_model
from tqdm import tqdm
import pdb


def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in tqdm(zip(HW[i_train], Ks[i_train], poses[i_train]), total=len(HW[i_train])):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max

'''
FourierGrid_compute_bbox_by_cam_frustrm_nerfpp 函数用于计算边界框。这个函数接受以下参数:

    cfg:一个包含配置信息的对象。
    HW:一个包含图像尺寸的列表。
    Ks:一个包含相机内参的列表。
    poses:一个包含相机位姿的列表。
    i_train:一个包含训练数据索引的列表。
    near_clip:一个表示近裁剪面距离的值。

函数的主要步骤如下:
    1. 初始化边界框的最小值(xyz_min)和最大值(xyz_max)。
    2. 遍历训练数据集中的相机位姿, 计算相机射线的原点(rays_o)和方向(rays_d)。
    3. 计算相机射线上位于近裁剪面处的点(pts)。
    4. 根据这些点的坐标, 更新边界框的最小值和最大值。
    5. 计算边界框的中心(center)和半径(radius)。
    6. 使用中心和半径, 重新计算边界框的最小值和最大值, 以得到一个覆盖所有相机中心的立方体。

函数返回计算出的边界框的最小值(xyz_min)和最大值(xyz_max)。这个边界框用于在场景中生成一个紧密的立方体, 
该立方体覆盖了所有相机中心。这有助于后续的渲染和训练过程, 因为它可以确保渲染的结果尽可能紧密地覆盖实际的场景范围。
'''

def FourierGrid_compute_bbox_by_cam_frustrm_nerfpp(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in tqdm(zip(HW[i_train], Ks[i_train], poses[i_train]), total=len(HW[i_train])):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0,1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max


def FourierGrid_compute_bbox_by_cam_frustrm_waymo(cfg, HW, Ks, poses, i_train, near_clip):
    xs, ys, zs = [], [], []
    for (H, W), K, c2w in tqdm(zip(HW[i_train], Ks[i_train], poses[i_train]), total=len(HW[i_train])):
        xs.append(c2w[:, 3][0].item())
        ys.append(c2w[:, 3][1].item())
        zs.append(c2w[:, 3][2].item())
    
    zmin, zmax = min(zs), max(zs)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x_extend = 0.05 # 0.05
    y_extend = 0.01
    z_extend = 0.01  # 0.01
    xyz_min = [xmin - x_extend, ymin - y_extend, zmin - z_extend]
    xyz_max = [xmax + x_extend, ymax + y_extend, zmax + z_extend]
    xyz_min, xyz_max = torch.tensor(xyz_min), torch.tensor(xyz_max)
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    
    # # manually set xyz_min and xyz_max
    # xyz_min = torch.tensor([-1.0, -1.0, -1.0])
    # xyz_max = torch.tensor([1.0, 1.0, 1.0])
    return xyz_min, xyz_max


def FourierGrid_compute_bbox_by_cam_frustrm_mega(cfg, HW, Ks, poses, i_train, near_clip):
    xs, ys, zs = [], [], []
    for (H, W), K, c2w in tqdm(zip(HW[i_train], Ks[i_train], poses[i_train]), total=len(HW[i_train])):
        xs.append(c2w[:, 3][0].item())
        ys.append(c2w[:, 3][1].item())
        zs.append(c2w[:, 3][2].item())
    zmin, zmax = min(zs), max(zs)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x_distance, y_distance, z_distance = abs(xmax - xmin), abs(ymax - ymin), abs(zmax - zmin)
    boundary_ratio = cfg.data.boundary_ratio
    xyz_min = [xmin - boundary_ratio*x_distance, ymin - boundary_ratio * y_distance, zmin - boundary_ratio * z_distance]
    xyz_max = [xmax + boundary_ratio*x_distance, ymax + boundary_ratio * y_distance, zmax + boundary_ratio * z_distance]
    xyz_min, xyz_max = torch.tensor(xyz_min), torch.tensor(xyz_max)
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,
                ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    verbose = args.block_num <= 1
    if verbose:
        print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.dataset_type == "waymo":
        xyz_min, xyz_max = FourierGrid_compute_bbox_by_cam_frustrm_waymo(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))
    elif cfg.data.dataset_type == "mega":
        xyz_min, xyz_max = FourierGrid_compute_bbox_by_cam_frustrm_mega(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))
    # todo: use another function to for tankstemple or llff
    # elif cfg.data.dataset_type == "nerfpp" or cfg.data.dataset_type == "tankstemple":
    elif cfg.data.dataset_type == "nerfpp" or cfg.data.dataset_type == "tankstemple" or cfg.model == 'FourierGrid':
        xyz_min, xyz_max = FourierGrid_compute_bbox_by_cam_frustrm_nerfpp(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))
    elif cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
                cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))
    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
                cfg, HW, Ks, poses, i_train, near, far)
    if verbose:
        print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
        print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
        print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max

@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres, device, args, cfg):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    # model, _, _ = load_existing_model(args, cfg, cfg.fine_train, model_path, device=device)
    model.to(device)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1-interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    if mask.max() > 0:
        active_xyz = dense_xyz[mask]
    else:
        print("Warning! No activated voxels found.")
        mask = (alpha > -1)
        active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max
