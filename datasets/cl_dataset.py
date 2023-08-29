import os
import numpy as np
from PIL import Image
from copy import deepcopy
import pyvista as pv

from torch.utils.data import Dataset

from utils.pose_utils import get_6dof_pose_label
from utils.preprocess import random_rotate_camera

class CLDataset(Dataset):
	def __init__(self, base_dir, dir_list, length, spacing, img_size, mesh_path, skip_prev_frame = False, transform_img = None, transform_pose = None):
		super(CLDataset, self).__init__()
		self.path_indices, self.samples = [], []
		fixed_dir_list = []
		self.dir_to_pose_dict = {}
		for this_dir in dir_list:
			this_dir = os.path.join(base_dir, this_dir)
			num_txt_files = len([t_path for t_path in os.listdir(this_dir) if t_path.endswith(".txt")])
			if num_txt_files == 0:
				for t_path in os.listdir(this_dir):
					if t_path.endswith(".npy"):
						full_path = os.path.join(this_dir, t_path)
						poses = np.load(full_path)
						fixed_dir_list.append((full_path, len(poses)))
						self.dir_to_pose_dict[full_path] = poses
			else:
				poses = []
				for img_idx in range(num_txt_files):
					this_pose = np.loadtxt(os.path.join(this_dir, f"{img_idx:06d}.txt"))
					poses.append(this_pose)
				fixed_dir_list.append((this_dir, num_txt_files))
				self.dir_to_pose_dict[this_dir] = np.stack(poses, 0)

		for this_dir, seq_len in fixed_dir_list:
			this_path_indices = []
			for i in range(seq_len):
				this_path_indices.append((this_dir, i))
			for i in range(seq_len):
				his_start = i - spacing if skip_prev_frame else i - 1
				indices = [i] + [max(his_start - j * spacing, 0) for j in range(length)]
				indices.reverse()
				self.samples.append([j + len(self.path_indices) for j in indices])
			self.path_indices.extend(this_path_indices)
		self.transform_img, self.transform_pose = transform_img, transform_pose
		self.img_size = img_size
		if mesh_path is None:
			self.plotter = None
		else:
			self.plotter = pv.Plotter(off_screen = True, window_size = (img_size, img_size))
			self.plotter.add_mesh(pv.read(mesh_path))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		input_img, input_pose = [], []
		for j, i in enumerate(self.samples[idx]):
			full_path, img_idx = self.path_indices[i]
			if self.plotter is None:
				img = np.load(os.path.join(full_path, f"{img_idx:06d}.npy"))
			else:
				img = None
			pose = self.dir_to_pose_dict[full_path][img_idx, ...]
			img, pose = random_rotate_camera(img, pose, self.img_size, self.plotter, False)
			pose = get_6dof_pose_label(pose)
			if self.transform_img is not None:
				img = self.transform_img(img)
			if self.transform_pose is not None:
				pose = self.transform_pose(pose, j == len(self.samples[idx]) - 1)
			input_img.append(img)
			input_pose.append(pose)
		input_img = np.stack(input_img)
		input_pose = np.stack(input_pose)
		return input_img, input_pose

class TestDataset(Dataset):
	def __init__(self, base_dir, dir_list, length, spacing, img_size, mesh_path, skip_prev_frame = False, transform_img = None, transform_pose = None):
		super(TestDataset, self).__init__()
		self.path_indices, self.samples = [], []

		fixed_dir_list = []
		self.dir_to_pose_dict = {}
		for this_dir in dir_list:
			this_dir = os.path.join(base_dir, this_dir)
			num_txt_files = len([t_path for t_path in os.listdir(this_dir) if t_path.endswith(".txt")])
			if num_txt_files == 0:
				for t_path in os.listdir(this_dir):
					if t_path.endswith(".npy"):
						full_path = os.path.join(this_dir, t_path)
						poses = np.load(full_path)
						fixed_dir_list.append((full_path, len(poses)))
						self.dir_to_pose_dict[full_path] = poses
			else:
				poses = []
				for img_idx in range(num_txt_files):
					this_pose = np.loadtxt(f"{img_idx:06d}.txt")
					poses.append(this_pose)
				fixed_dir_list.append((this_dir, num_txt_files))
				self.dir_to_pose_dict[this_dir] = np.stack(poses, 0)

		for this_dir, seq_len in fixed_dir_list:
			this_path_indices = []
			for i in range(seq_len):
				this_path_indices.append((this_dir, i))
			for i in range(seq_len):
				his_start = i - spacing if skip_prev_frame else i - 1
				indices = [i] + [max(his_start - j * spacing, 0) for j in range(length)]
				indices.reverse()
				indices = [j + len(self.path_indices) for j in indices]
				use_hisenc = 0 if his_start - length * spacing < 0 else 1
				self.samples.append((indices, use_hisenc))
			self.path_indices.extend(this_path_indices)
		self.transform_img, self.transform_pose = transform_img, transform_pose
		self.img_size = img_size
		if mesh_path is None:
			self.plotter = None
		else:
			self.plotter = pv.Plotter(off_screen = True, window_size = (img_size, img_size))
			self.plotter.add_mesh(pv.read(mesh_path))


	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		input_img, input_pose, his_indices = [], [], []
		indices, use_hisenc = self.samples[idx]
		for j, i in enumerate(indices):
			full_path, img_idx = self.path_indices[i]
			if self.plotter is None:
				img = np.load(os.path.join(full_path, f"{img_idx:06d}.npy"))
			else:
				img = None
			pose = self.dir_to_pose_dict[full_path][img_idx, ...]
			img, pose = random_rotate_camera(img, pose, self.img_size, self.plotter, False)
			pose = get_6dof_pose_label(pose)

			if self.transform_img is not None:
				img = self.transform_img(img)
			if self.transform_pose is not None:
				pose = self.transform_pose(pose, j == len(self.samples[idx]) - 1)
			input_img.append(img)
			input_pose.append(pose)
			his_indices.append(i)
		input_img = np.stack(input_img)
		return input_img, input_pose[-1], np.array(indices[ : -1]), indices[-1], use_hisenc
