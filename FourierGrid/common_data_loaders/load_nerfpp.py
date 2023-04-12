'''
Modify from
https://github.com/Kai-46/nerfplusplus/blob/master/data_loader_split.py
'''
import os
import cv2
import pdb
import glob
import scipy
import imageio
import numpy as np
import torch
from skimage.filters import roberts, prewitt

########################################################################################################################
# camera coordinate system: x-->right, y-->down, z-->scene (opencv/colmap convention)
# poses is camera-to-world
########################################################################################################################
def find_files(dir, exts):
    if os.path.isdir(dir):
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []


def load_data_split(split_dir, skip=1, try_load_min_depth=True, only_img_files=False,
                    training_ids=None):

    def parse_txt(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

    if only_img_files:
        img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
        return img_files

    # camera parameters files
    intrinsics_files = find_files('{}/intrinsics'.format(split_dir), exts=['*.txt'])
    pose_files = find_files('{}/pose'.format(split_dir), exts=['*.txt'])

    intrinsics_files = intrinsics_files[::skip]
    pose_files = pose_files[::skip]
    cam_cnt = len(pose_files)

    # img files
    img_files = find_files('{}/rgb'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(img_files) > 0:
        img_files = img_files[::skip]
        assert(len(img_files) == cam_cnt)
    else:
        img_files = [None, ] * cam_cnt

    # mask files
    mask_files = find_files('{}/mask'.format(split_dir), exts=['*.png', '*.jpg'])
    if len(mask_files) > 0:
        mask_files = mask_files[::skip]
        assert(len(mask_files) == cam_cnt)
    else:
        mask_files = [None, ] * cam_cnt

    # min depth files
    mindepth_files = find_files('{}/min_depth'.format(split_dir), exts=['*.png', '*.jpg'])
    if try_load_min_depth and len(mindepth_files) > 0:
        mindepth_files = mindepth_files[::skip]
        assert(len(mindepth_files) == cam_cnt)
    else:
        mindepth_files = [None, ] * cam_cnt
    
    # sample by training ids
    if training_ids is not None:
        final_training_ids = []
        for idx, ele in enumerate(intrinsics_files):
            if int(ele.split("/")[-1].replace(".txt", "")) in training_ids:
                final_training_ids.append(idx)
        training_ids = final_training_ids
        training_ids = [id - 1 for id in training_ids]  # image id start with 1
        intrinsics_files = [intrinsics_files[id] for id in training_ids]
        pose_files = [pose_files[id] for id in training_ids]
        img_files = [img_files[id] for id in training_ids]
        mask_files = [mask_files[id] for id in training_ids]
        mindepth_files = [mindepth_files[id] for id in training_ids]
    return intrinsics_files, pose_files, img_files, mask_files, mindepth_files

'''
这个 rerotate_poses 函数的主要目的是对输入的相机位姿(poses)和渲染位姿(render_poses)进行重新旋转，以便将场景对齐到一个统一的坐标系。函数的步骤如下：

    1. 首先，创建相机位姿的副本，以避免修改原始数据。
    2. 计算位姿中心 (centroid)，并将位姿的平移分量(第三列)减去中心，使位姿围绕原点对齐。
    3. 计算位姿平移分量的主成分分析 (PCA)，找到具有最小特征值的主成分向量 (cams_up)。这个向量表示场景的“向上”方向。
    4. 如果 cams_up 向量的 y 分量小于零，将其取反，确保向量的 y 分量为正。
    5. 使用 scipy.spatial.transform.Rotation.align_vectors 方法，找到将 cams_up 向量对齐到 [0, -1, 0] 向量的旋转矩阵(R)。这里 [0, -1, 0] 表示目标向上方向。
    6. 将旋转矩阵 R 应用于相机位姿和渲染位姿的旋转和平移分量。这会将场景的向上方向对齐到 [0, -1, 0] 向量。
    7. 将之前减去的位姿中心加回到平移分量中，将位姿恢复到原始尺度。
    8. 对渲染位姿执行类似的操作，包括减去中心、应用旋转矩阵和加回中心。
    9. 返回经过重新旋转的相机位姿和渲染位姿。

通过这个函数，相机位姿和渲染位姿都会被重新旋转，使得场景的向上方向与目标方向([0, -1, 0])对齐。这有助于在不同数据集之间保持一致性，从而改善训练和评估过程。
'''

def rerotate_poses(poses, render_poses):
    poses = np.copy(poses)
    centroid = poses[:,:3,3].mean(0)

    poses[:,:3,3] = poses[:,:3,3] - centroid

    # Find the minimum pca vector with minimum eigen value
    x = poses[:,:3,3]
    mu = x.mean(0)
    cov = np.cov((x-mu).T)
    ev , eig = np.linalg.eig(cov)
    cams_up = eig[:,np.argmin(ev)]
    if cams_up[1] < 0:
        cams_up = -cams_up

    # Find rotation matrix that align cams_up with [0,1,0]
    R = scipy.spatial.transform.Rotation.align_vectors(
            [[0,-1,0]], cams_up[None])[0].as_matrix()

    # Apply rotation and add back the centroid position
    poses[:,:3,:3] = R @ poses[:,:3,:3]
    poses[:,:3,[3]] = R @ poses[:,:3,[3]]
    poses[:,:3,3] = poses[:,:3,3] + centroid
    render_poses = np.copy(render_poses)
    render_poses[:,:3,3] = render_poses[:,:3,3] - centroid
    render_poses[:,:3,:3] = R @ render_poses[:,:3,:3]
    render_poses[:,:3,[3]] = R @ render_poses[:,:3,[3]]
    render_poses[:,:3,3] = render_poses[:,:3,3] + centroid
    return poses, render_poses

'''
这段代码定义了一个名为 load_nerfpp_data 的函数，用于加载 NeRF++ 数据集。NeRF++ 是一种神经网络,
用于生成3D场景的视图。函数的输入包括数据的基本目录 (basedir)、一个布尔值 (rerotate) 以及训练数据的可选ID列表 (training_ids)。
这个函数的主要目的是从给定的目录加载图像、相机内参、相机位姿以及生成轨迹，并将它们整理成便于后续处理的数据结构。

以下是代码主要步骤的简要概述：

    1.  加载训练和测试数据集的相机内参(train_K 和 test_K)、相机位姿(train_camera2world 和 test_camera2world)以及图像路径(train_image_path 和 test_image_path)。
    2.  确保训练和测试数据集的元素数量一致。
    3.  确定训练和测试数据集的索引列表 (i_split)。
    4.  加载相机内参，并确保所有图像共享相同的内参。
    5.  加载训练和测试数据集的相机位姿。
    6.  加载训练和测试数据集的图像，并将它们归一化到 [0,1] 范围。
    7.  将所有图像和相机位姿堆叠到一个数组中。
    8.  从相机内参中计算焦距 (focal)。
    9.  生成电影轨迹 (render_poses)。
    10. 如果 rerotate 为真，则重新旋转相机位姿。
    11. 返回加载的数据，包括图像、相机位姿、渲染位姿、图像维度、相机内参和数据集划分索引。

这个函数主要用于加载和整理 NeRF++ 数据集，使其可以用于后续的训练和评估任务。
'''

def load_nerfpp_data(cfg, basedir, rerotate=True, training_ids=None):
    train_K, train_camera2world, train_image_path = load_data_split(os.path.join(basedir, 'train'), training_ids=training_ids)[:3]
    assert len(train_image_path) > 0, f"Images are not found in {basedir}"
    test_K, test_camera2world, test_image_path = load_data_split(os.path.join(basedir, 'test'))[:3]
    assert len(train_K) == len(train_camera2world) and len(train_K) == len(train_image_path)
    assert len(test_K) == len(test_camera2world) and len(test_K) == len(test_image_path)

    # Determine split id list
    i_split = [[], []]
    i = 0
    for _ in train_camera2world:
        i_split[0].append(i)
        i += 1
    for _ in test_camera2world:
        i_split[1].append(i)
        i += 1

    # Load camera intrinsics. Assume all images share a intrinsic.
    K_flatten = np.loadtxt(train_K[0])
    for path in train_K:
        assert np.allclose(np.loadtxt(path), K_flatten)
    for path in test_K:
        assert np.allclose(np.loadtxt(path), K_flatten)
    K = K_flatten.reshape(4,4)[:3,:3]

    # Load camera poses
    poses = []
    for path in train_camera2world:
        poses.append(np.loadtxt(path).reshape(4,4))
    for path in test_camera2world:
        poses.append(np.loadtxt(path).reshape(4,4))

    # Load images
    imgs, edgeimags = [], []
    for path in train_image_path + test_image_path:
        img = imageio.imread(path) / 255.
        imgs.append(img)
        color_image = cv2.imread(path)
        color_image = color_image.astype(np.uint8)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        if cfg.edgeType == "canny":
            edges_gray_image = cv2.Canny(gray_image, 100, 200)
            edgeimg = edges_gray_image.astype(np.float32) / 255.
            edgeimags.append(edgeimg)
        elif cfg.edgeType == "sobel":
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_edges = np.hypot(sobel_x, sobel_y)
            _, edges_gray_image = cv2.threshold(sobel_edges, 127, 255, cv2.THRESH_BINARY)
            edgeimg = edges_gray_image.astype(np.float32) / 255.
            edgeimags.append(edgeimg)
        elif cfg.edgeType == "laplacian":
            laplacian_edges = cv2.Laplacian(gray_image, cv2.CV_64F)
            _, edges_gray_image = cv2.threshold(laplacian_edges, 127, 255, cv2.THRESH_BINARY)
            edgeimg = edges_gray_image.astype(np.float32) / 255.
            edgeimags.append(edgeimg)
        elif cfg.edgeType == "roberts":
            roberts_edges = roberts(gray_image)
            _, edges_gray_image = cv2.threshold(roberts_edges, 0.1, 255, cv2.THRESH_BINARY)
            edgeimg = edges_gray_image.astype(np.float32) / 255.
            edgeimags.append(edgeimg)
        elif cfg.edgeType == "prewitt":
            prewitt_edges = prewitt(gray_image)
            _, edges_gray_image = cv2.threshold(prewitt_edges, 0.1, 255, cv2.THRESH_BINARY)    
            edgeimg = edges_gray_image.astype(np.float32) / 255.
            edgeimags.append(edgeimg)

    # Bundle all data
    imgs = np.stack(imgs, 0) # (N, H, W, 3)
    edgeimags = np.stack(edgeimags, 0) # (N, H, W)
    
    poses = np.stack(poses, 0)
    i_split.append(i_split[1])
    H, W = imgs.shape[1:3]
    focal = K[[0,1], [0,1]].mean()

    # Generate movie trajectory
    render_poses_path = sorted(glob.glob(os.path.join(basedir, 'camera_path', 'pose', '*txt')))
    render_poses = []
    for path in render_poses_path:
        render_poses.append(np.loadtxt(path).reshape(4,4))
    render_poses = np.array(render_poses)
    render_K = np.loadtxt(glob.glob(os.path.join(basedir, 'camera_path', 'intrinsics', '*txt'))[0]).reshape(4,4)[:3,:3]
    render_poses[:,:,0] *= K[0,0] / render_K[0,0]
    render_poses[:,:,1] *= K[1,1] / render_K[1,1]
    if rerotate:
        poses, render_poses = rerotate_poses(poses, render_poses)
    render_poses = torch.Tensor(render_poses)
    return imgs, poses, render_poses, [H, W, focal], K, i_split, edgeimags
