import json
import numpy as np
from torchvision import transforms

def get_img_transform(data_stats_path, img_size):
	stats = json.load(open(data_stats_path))
	img_mean, img_std = [stats["img_mean"]] * 3, [stats["img_std"]] * 3
	if img_size != -1:
		res = transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(img_mean, img_std)])
	else:
		res = transforms.Compose([transforms.ToTensor(), transforms.Normalize(img_mean, img_std)])
	return res

def get_pose_transforms(data_stats_path, hispose_noise):
	stats = json.load(open(data_stats_path))
	pose_mean, pose_std = np.array(stats["pose_mean"]), np.array(stats["pose_std"])
	def trans(x, true_pose):
		x = (x - pose_mean) / pose_std
		if not true_pose:
			x = x + np.random.randn(*x.shape) * hispose_noise
		return x
	inv_trans = lambda x : x * pose_std + pose_mean
	return trans, inv_trans
