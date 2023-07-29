import json
import torch
import numpy as np
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import compute_rotation_quaternion

def get_img_transform(data_stats_path, img_size, modality):
	stats = json.load(open(data_stats_path))
	img_mean, img_std = [stats["img_mean"]] * 3, [stats["img_std"]] * 3
	trans_list = [transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor()]
	if modality == "SFS":
		W, B = stats["sfs_intensity_w"] / 255, stats["sfs_intensity_b"] / 255
		def trans_intensity(x):
			x = torch.maximum(torch.minimum(x * W + B, torch.tensor(1.0)), torch.tensor(0.0))
		trans_list.append(transforms.Lambda(trans_intensity))
	elif modality == "mesh":
		pass
	else:
		raise NotImplementedError
	trans_list.append(transforms.Normalize(img_mean, img_std))
	return transforms.Compose(trans_list)

def get_pose_transforms(data_stats_path, hispose_noise, modality):
	stats = json.load(open(data_stats_path))
	pose_mean, pose_std = np.array(stats["pose_mean"]), np.array(stats["pose_std"])
	def trans_norm(x, true_pose):
		x = (x - pose_mean) / pose_std
		if not true_pose:
			x = x + np.random.randn(*x.shape) * hispose_noise
		return x
	if modality == "SFS":
		ct_origin, em_origin = np.array(stats["ct_origin"]), np.array(stats["em_origin"])
		def trans_em(x):
			return compute_rotation_quaternion(ct_origin, R.from_quat(x).apply(em_origin))
		trans = lambda x : trans_norm(trans_em(x))
	elif modality == "mesh":
		trans = trans_norm
	else:
		raise NotImplementedError
	inv_trans = lambda x : x * pose_std + pose_mean
	return trans, inv_trans
