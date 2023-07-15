import json
import numpy as np
from torchvision import transforms

def get_img_transform(args):
	stats = json.load(open(args.data_stats))
	img_mean, img_std = [stats["img_mean"]] * 3, [stats["img_std"]] * 3
	if args.img_size != -1:
		res = transforms.Compose([transforms.Resize(args.img_size), transforms.CenterCrop(args.img_size), transforms.ToTensor(), transforms.Normalize(img_mean, img_std)])
	else:
		res = transforms.Compose([transforms.ToTensor(), transforms.Normalize(img_mean, img_std)])
	return res

def get_pose_transforms(args):
	if args.no_normalise:
		return lambda x : x, lambda x : x
	stats = json.load(open(args.data_stats))
	pose_mean, pose_std = np.array(stats["pose_mean"]), np.array(stats["pose_std"])
	trans = lambda x : (x - pose_mean) / pose_std
	inv_trans = lambda x : x * pose_std + pose_mean
	return trans, inv_trans
