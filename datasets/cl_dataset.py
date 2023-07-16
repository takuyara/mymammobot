import os
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from utils.pose_utils import get_6dof_pose_label

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

def load_single_file(path, idx):
	this_file = f"{idx:06d}.png"
	this_img = Image.open(os.path.join(path, this_file)).convert("RGB")
	this_file = f"{idx:06d}.txt"
	this_pose = np.loadtxt(os.path.join(path, this_file))
	this_pose = get_6dof_pose_label(this_pose)
	return this_img, this_pose

class CLDataset(Dataset):
	def __init__(self, base_dir, dir_list, length, spacing, skip_prev_frame = False, transform_img = None, transform_pos = None):
		super(CLDataset, self).__init__()
		self.path_indices, self.samples = [], []
		for this_dir in dir_list:
			this_dir = os.path.join(base_dir, this_dir)
			this_path_indices = []
			for i in range(len(os.listdir(this_dir)) // 2):
				this_path_indices.append((this_dir, i))
			for i in range(len(this_path_indices)):
				his_start = i - spacing if skip_prev_frame else i - 1
				indices = [i] + [max(his_start - j * spacing, 0) for j in range(length)]
				indices.reverse()
				self.samples.append([j + len(self.path_indices) for j in indices])
			self.path_indices.extend(this_path_indices)
		self.transform_img, self.transform_pos = transform_img, transform_pos
	
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		input_img, input_pos = [], []
		for j, i in enumerate(self.samples[idx]):
			img, pos = load_single_file(*self.path_indices[i])
			if self.transform_img is not None:
				img = self.transform_img(img)
			if self.transform_pos is not None:
				pos = self.transform_pos(pos, j == len(self.samples[idx]) - 1)
			img, pos = torch.tensor(img), torch.tensor(pos)
			input_img.append(img)
			input_pos.append(pos)
		input_img = torch.stack(input_img)
		input_pos = torch.stack(input_pos)
		return input_img, input_pos

class TestDataset(Dataset):
	def __init__(self, base_dir, dir_list, length, spacing, skip_prev_frame = False, transform_img = None, transform_pos = None):
		super(TestDataset, self).__init__()
		self.path_indices, self.samples = [], []
		for this_dir in dir_list:
			this_dir = os.path.join(base_dir, this_dir)
			this_path_indices = []
			for i in range(len(os.listdir(this_dir)) // 2):
				this_path_indices.append((this_dir, i))
			for i in range(len(this_path_indices)):
				his_start = i - spacing if skip_prev_frame else i - 1
				indices = [i] + [max(his_start - j * spacing, 0) for j in range(length)]
				indices.reverse()
				indices = [j + len(self.path_indices) for j in indices]
				use_hisenc = 0 if his_start - length * spacing < 0 else 1
				self.samples.append((indices, use_hisenc))
			self.path_indices.extend(this_path_indices)
		self.transform_img, self.transform_pos = transform_img, transform_pos

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		input_img, input_pos, his_indices = [], [], []
		indices, use_hisenc = self.samples[idx]
		for j, i in enumerate(indices):
			img, pos = load_single_file(*self.path_indices[i])
			if self.transform_img is not None:
				img = self.transform_img(img)
			if self.transform_pos is not None:
				pos = self.transform_pos(pos, j == len(self.samples[idx]) - 1)
			img, pos = torch.tensor(img), torch.tensor(pos)
			input_img.append(img)
			input_pos.append(pos)
			his_indices.append(i)
		input_img = torch.stack(input_img)
		return input_img, input_pos[-1], np.array(indices[ : -1]), indices[-1], use_hisenc
