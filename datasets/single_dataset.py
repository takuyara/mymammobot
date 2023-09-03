import os
import numpy as np
import pyvista as pv

import torch
from torch.utils.data import Dataset

from utils.pose_utils import get_6dof_pose_label
from utils.preprocess import random_rotate_camera
from utils.misc import randu_gen

class SingleImageDataset(Dataset):
	def __init__(self, base_dir, dir_list, img_size, mesh_path, convert_to_clbase, transform_img, transform_pose):
		super(SingleImageDataset, self).__init__()
		self.samples = []
		angle_gen = randu_gen(0, 360)
		for this_dir in dir_list:
			this_dir = os.path.join(base_dir, this_dir)
			for path in os.listdir(this_dir):
				if path.endswith(".txt"):
					self.samples.append((this_dir, int(path.replace(".txt", ""))))
		self.transform_img, self.transform_pose = transform_img, transform_pose
		self.img_size = img_size
		#self.zoom_gen = randu_gen(0.9, 1.1)
		self.zoom_gen = lambda : 1
		if mesh_path is None:
			self.plotter = None
		else:
			self.plotter = pv.Plotter(off_screen = True, window_size = (img_size, img_size))
			self.plotter.add_mesh(pv.read(mesh_path))

		self.convert_to_clbase = convert_to_clbase
	
	def trans(self, x):
		if self.cl_ref is None:
			return x
		cl_dists = np.sum((x[...,  : 3] - self.cl_ref) ** 2, axis = -1)
		return int(self.cl_ref_labels[np.argmin(cl_dists)])
	
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		p, i = self.samples[idx]
		if self.plotter is None:
			img = np.load(os.path.join(p, f"{i:06d}.npy"))
		else:
			img = None
		pose = np.loadtxt(os.path.join(p, f"{i:06d}.txt"))
		img, pose = random_rotate_camera(img, pose, self.img_size, self.plotter, 0, zoom_scale = self.zoom_gen())
		pose = get_6dof_pose_label(pose)
		if self.convert_to_clbase:
			cl_base_pose = np.loadtxt(os.path.join(p, f"{i:06d}_clbase.txt")).ravel()
			pose = int(cl_base_pose[0])
		
		if self.transform_img is not None:
			img = self.transform_img(img)
		if self.transform_pose is not None:
			pose = self.transform_pose(pose, True)
		return img, pose
