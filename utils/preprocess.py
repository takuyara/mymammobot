import json
import torch
import numpy as np
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter

from ds_gen.rotatable_single_images import rotate_and_crop
from ds_gen.depth_map_generation import get_depth_map
from utils.pose_utils import compute_rotation_quaternion, get_3dof_quat, revert_quat, camera_pose_to_train_pose
from utils.geometry import rotate_single_vector, arbitrary_perpendicular_vector

def random_rotate_camera(img, pose, img_size, plotter = None, rotatable = True):
	position, orientation = pose[0, ...], pose[1, ...]
	if rotatable:
		deg = np.random.rand() * 360
		up = rotate_single_vector(arbitrary_perpendicular_vector(orientation), orientation, deg)
	else:
		deg = 0
		up = pose[2, ...]
	if plotter is None:
		img = rotate_and_crop(img, deg, img_size)
	else:
		img = get_depth_map(plotter, position, orientation, up)
	img = np.nan_to_num(img, nan = np.nanmax(img))
	pose = camera_pose_to_train_pose(position, orientation, up)
	return img, pose

def get_img_transform(data_stats_path, method = "norm"):
	stats = json.load(open(data_stats_path))
	if method in ["sfs2mesh", "mesh2sfs", "sfs", "mesh"]:
		if method in ["sfs2mesh", "mesh"]:
			img_mean, img_std = stats["mesh_mean"], stats["mesh_std"]
		else:
			img_mean, img_std = stats["sfs_mean"], stats["sfs_std"]
		if method == "sfs2mesh":
			_w, _b = stats["sfs2mesh_weight"], stats["sfs2mesh_bias"]
		elif method == "mesh2sfs":
			_w, _b = stats["mesh2sfs_weight"], stats["mesh2sfs_bias"]
			kernel_size = stats["mesh2sfs_kernel"]
			radius = (kernel_size - 1) // 2
			sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
		else:
			_w, _b = 1, 0
		def reshape_n_norm(img):
			if method == "mesh2sfs":
				img = gaussian_filter(img, sigma = sigma, radius = radius)
			img = torch.tensor(img).float().unsqueeze(0)
			img = img * _w + _b
			img = (img - img_mean) / img_std
			return img			
		return reshape_n_norm
	elif method == "quantile":
		def img_to_quantile(img):
			orig_shape = img.shape
			img = img.ravel()
			q = np.argsort(img)
			q = np.argsort(q)
			q = q / len(q)
			q = q.reshape(orig_shape)
			q = torch.tensor(q).float().unsqueeze(0)
			return q
		return img_to_quantile
	elif method == "hist_simple":
		def img_to_hist_simple(img, bins = 30):
			img = (img - img.min()) / (img.max() - img.min())
			img = np.floor(img * bins) / bins
			return torch.tensor(img).float().unsqueeze(0)
		return img_to_hist_simple
	elif method == "hist_complex":
		def img_to_hist_complex(img, bins = 30):
			orig_shape = img.shape
			img = (img - img.min()) / (img.max() - img.min())
			img_hist_indices = np.minimum(np.floor(img * bins).astype(int), bins - 1)
			img_hist_heights = np.histogram(img.ravel(), bins = 30, density = True)[0]
			hist_peak_idx = np.argmax(img_hist_heights)
			img_hist_heights = img_hist_heights / img_hist_heights[hist_peak_idx]
			#print("Prev: ", [round(x, 1) for x in img_hist_heights])
			for j in range(len(img_hist_heights)):
				i = len(img_hist_heights) - j - 1
				if i < hist_peak_idx:
					img_hist_heights[i] += 1
				if j > 0:
					img_hist_heights[i] = max(img_hist_heights[i], img_hist_heights[i + 1])
			#print("Succ: ", [round(x, 1) for x in img_hist_heights])
			labels = img_hist_heights[img_hist_indices]
			labels = labels.reshape(orig_shape)
			mean, std = stats["hist_complex_mean"], stats["hist_complex_std"]
			labels = (labels - mean) / std
			return torch.tensor(labels).float().unsqueeze(0)
		return img_to_hist_complex

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
