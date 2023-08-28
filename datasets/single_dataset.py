import os
import numpy as np
import pyvista as pv

import torch
from torch.utils.data import Dataset

from utils.pose_utils import get_6dof_pose_label
from utils.preprocess import random_rotate_camera

class SingleImageDataset(Dataset):
	def __init__(self, base_dir, dir_list, img_size, mesh_path, rotatable, pack_data_size = 3, transform_img = None, transform_pose = None):
		super(SingleImageDataset, self).__init__()
		self.samples = []
		for this_dir in dir_list:
			this_dir = os.path.join(base_dir, this_dir)
			for i in range(len(os.listdir(this_dir)) // pack_data_size):
				self.samples.append((this_dir, i))
		self.transform_img, self.transform_pose = transform_img, transform_pose
		self.img_size = img_size
		self.rotatable = rotatable
		if mesh_path is None:
			self.plotter = None
		else:
			self.plotter = pv.Plotter(off_screen = True, window_size = (img_size, img_size))
			self.plotter.add_mesh(pv.read(mesh_path))
	
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		p, i = self.samples[idx]
		if self.plotter is None:
			img = np.load(os.path.join(p, f"{i:06d}.npy"))
		else:
			img = None
		pose = np.loadtxt(os.path.join(p, f"{i:06d}.txt"))
		img, pose = random_rotate_camera(img, pose, self.img_size, self.plotter, self.rotatable)
		pose = get_6dof_pose_label(pose)
		
		if self.transform_img is not None:
			img = self.transform_img(img)
		if self.transform_pose is not None:
			pose = self.transform_pose(pose, True)
		return img, pose
