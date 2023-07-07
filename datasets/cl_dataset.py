import os
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

class CLDataset(Dataset):
	def __init__(self, base_dir, dir_list, length, spacing, transform_img = None, transform_pos = None):
		super(CLDataset, self).__init__()
		all_imgs, all_poses = [], []
		all_samples = []
		for this_dir in dir_list:
			this_dir = os.path.join(base_dir, this_dir)
			file_imgs, file_poses = [], []
			for i in range(len(os.listdir(this_dir)) // 2):
				this_file = f"frame-{i:06d}.color.png"
				this_img = Image.open(os.path.join(this_dir, this_file))
				file_imgs.append(deepcopy(this_img))
				this_file = f"frame-{i:06d}.pose.txt"
				this_pose = np.loadtxt(os.path.join(this_dir, this_file))
				this_trans = this_pose.reshape(4, 4)[ : 3, 3]
				this_rot = np.zeros(3)
				this_pose = np.concatenate([this_trans, this_rot])
				file_poses.append(this_pose)
			for i in range(len(file_imgs) - 1):
				indices = [i + 1] + [max(i - j * spacing, 0) for j in range(length)]
				indices.reverse()
				xx = [j + len(all_imgs) for j in indices]
				all_samples.append([j + len(all_imgs) for j in indices])
			all_imgs.extend(file_imgs)
			all_poses.extend(file_poses)
		self.imgs, self.poses, self.samples = all_imgs, all_poses, all_samples
		self.transform_img, self.transform_pos = transform_img, transform_pos
	
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		input_img, input_pos = [], []
		for i in self.samples[idx]:
			img, pos = self.imgs[i], self.poses[i]
			if self.transform_img is not None:
				img = self.transform_img(img)
			if self.transform_pos is not None:
				pos = self.transform_pos(pos)
			img, pos = torch.tensor(img), torch.tensor(pos)
			input_img.append(img)
			input_pos.append(pos)
		input_img = torch.stack(input_img)
		input_pos = torch.stack(input_pos)
		return input_img, input_pos
