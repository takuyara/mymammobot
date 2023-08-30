import os
import json
import cv2
import numpy as np
from utils.arguments import get_args
from utils.file_utils import get_dir_list
from utils.pose_utils import get_6dof_pose_label
from utils.preprocess import get_img_transform
from tqdm import tqdm

def compute_pose_stats(base_dir, dir_list):
	all_poses = []
	for this_dir in dir_list:
		this_dir = os.path.join(base_dir, this_dir)
		this_path_indices = []
		for i in range(len(os.listdir(this_dir)) // 3):
			this_pose = np.loadtxt(os.path.join(this_dir, f"{i:06d}.txt"))
			all_poses.append(this_pose[0, ...])
	all_poses = np.array(all_poses)
	return np.mean(all_poses, axis = 0), np.std(all_poses, axis = 0)

def compute_img_stats(base_dir, dir_list):
	sum_pix, sum_var, n_pix = 0, 0, 0
	trans = get_img_transform("data_stats.json", "hist_even_more_complex", 2, True)
	for this_dir in dir_list:
		this_dir = os.path.join(base_dir, this_dir)
		this_path_indices = []
		for i in tqdm(range(len(os.listdir(this_dir)) // 3)):
			#img = cv2.imread(os.path.join(this_dir, this_file), cv2.IMREAD_GRAYSCALE).reshape(-1)
			img = trans(np.load(os.path.join(this_dir,  f"{i:06d}.npy"))).numpy().ravel()
			sum_pix += np.sum(img)
			n_pix += img.shape[0]
	img_mean = sum_pix / n_pix
	for this_dir in dir_list:
		this_dir = os.path.join(base_dir, this_dir)
		this_path_indices = []
		for i in tqdm(range(len(os.listdir(this_dir)) // 3)):
			#img = cv2.imread(os.path.join(this_dir, this_file), cv2.IMREAD_GRAYSCALE).reshape(-1)
			img = trans(np.load(os.path.join(this_dir,  f"{i:06d}.npy"))).numpy().ravel()
			sum_var += np.sum((img - img_mean) ** 2)
	img_std = (sum_var / n_pix) ** 0.5
	return img_mean, img_std

def main():
	args = get_args()
	#pose_mean, pose_std = compute_pose_stats(args.base_dir, get_dir_list(args.train_split))
	img_mean, img_std = compute_img_stats("./virtual_dataset", ["single_image_reduced/train"])
	print(img_mean, img_std)
	#stats = {"img_mean": img_mean.tolist(), "img_std": img_std.tolist(), "pose_mean": pose_mean.tolist() + [0, 0, 0], "pose_std": pose_std.tolist() + [1, 1, 1]}
	#json.dump(stats, open("data_stats.json", "w"))

if __name__ == '__main__':
	main()
