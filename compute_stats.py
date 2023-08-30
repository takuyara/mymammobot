import os
import json
import cv2
import numpy as np
import pyvista as pv
from utils.arguments import get_args
from utils.file_utils import get_dir_list
from utils.pose_utils import get_6dof_pose_label
from utils.preprocess import get_img_transform
from utils.geometry import arbitrary_perpendicular_vector
from ds_gen.depth_map_generation import get_depth_map
from tqdm import tqdm

def compute_pose_stats(base_dir, dir_list):
	all_poses = []
	for this_dir in dir_list:
		this_dir = os.path.join(base_dir, this_dir)
		for this_path in os.listdir(this_dir):
			if this_path.endswith(".txt"):
				this_pose = np.loadtxt(os.path.join(this_dir, this_path))
				all_poses.append(this_pose[0, ...])
	all_poses = np.array(all_poses)
	return np.mean(all_poses, axis = 0), np.std(all_poses, axis = 0)

def compute_img_stats(p, base_dir, dir_list):
	sum_pix, sum_var, n_pix = 0, 0, 0
	trans = get_img_transform("data_stats.json", "quantile", 1, False)
	for this_dir in dir_list:
		this_dir = os.path.join(base_dir, this_dir)
		for this_path in os.listdir(this_dir):
			if this_path.endswith(".txt"):
				this_pose = np.loadtxt(os.path.join(this_dir, this_path))
				img = get_depth_map(p, this_pose[0], this_pose[1], arbitrary_perpendicular_vector(this_pose[1]))
				img = trans(img).numpy().ravel()
				sum_pix += np.sum(img)
				n_pix += img.shape[0]
	img_mean = sum_pix / n_pix
	for this_dir in dir_list:
		this_dir = os.path.join(base_dir, this_dir)
		for this_path in os.listdir(this_dir):
			if this_path.endswith(".txt"):
				this_pose = np.loadtxt(os.path.join(this_dir, this_path))
				img = get_depth_map(p, this_pose[0], this_pose[1], arbitrary_perpendicular_vector(this_pose[1]))
				img = trans(img).numpy().ravel()
				sum_var += np.sum((img - img_mean) ** 2)
	img_std = (sum_var / n_pix) ** 0.5
	return img_mean, img_std

def main():
	args = get_args()
	p = pv.Plotter(off_screen = True, window_size = (args.img_size, args.img_size))
	p.add_mesh(pv.read(args.mesh_path))
	#pose_mean, pose_std = compute_pose_stats(args.base_dir, get_dir_list(args.train_split))
	img_mean, img_std = compute_img_stats(p, "./virtual_dataset", ["single_image_reduced/train"])
	print(img_mean, img_std)
	#stats = {"img_mean": img_mean.tolist(), "img_std": img_std.tolist(), "pose_mean": pose_mean.tolist() + [0, 0, 0], "pose_std": pose_std.tolist() + [1, 1, 1]}
	#json.dump(stats, open("data_stats.json", "w"))

if __name__ == '__main__':
	main()
