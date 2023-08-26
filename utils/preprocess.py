import json
import torch
import numpy as np
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

from ds_gen.rotatable_single_images import rotate_and_crop
from ds_gen.depth_map_generation import get_depth_map
from utils.pose_utils import compute_rotation_quaternion, get_3dof_quat, revert_quat, camera_pose_to_train_pose
from utils.geometry import rotate_single_vector, arbitrary_perpendicular_vector

def random_rotate_camera(img, position, orientation, img_size, plotter = None):
	deg = np.random.rand() * 360
	up = rotate_single_vector(arbitrary_perpendicular_vector(orientation), orientation, deg)
	if plotter is None:
		img = rotate_and_crop(img, deg, img_size)
	else:
		img = get_depth_map(plotter, position, orientation, up)
	pose = camera_pose_to_train_pose(position, orientation, up)
	return img, pose

def get_img_transform(data_stats_path, img_size, modality, train = False):
	stats = json.load(open(data_stats_path))
	img_mean, img_std = stats["img_mean"], stats["img_std"]
	def reshape_n_norm(img):
		img = torch.tensor(img).float().unsqueeze(0)
		img = (img - img_mean) / img_std
		return img
	return reshape_n_norm

def get_pose_transforms(data_stats_path, hispose_noise, modality):
	stats = json.load(open(data_stats_path))
	pose_mean, pose_std = np.array(stats["pose_mean"]), np.array(stats["pose_std"])
	def trans_norm(x, true_pose):
		x = (x - pose_mean) / pose_std
		if not true_pose:
			x = x + np.random.randn(*x.shape) * hispose_noise
		return x
	trans = trans_norm
	inv_trans = lambda x : x * pose_std + pose_mean
	return trans, inv_trans
