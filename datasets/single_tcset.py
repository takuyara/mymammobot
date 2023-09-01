import os
import numpy as np
import pyvista as pv

import torch
from torch.utils.data import Dataset

from utils.preprocess import load_img_pose

class SingleTCDataset(Dataset):
	def __init__(self, base_dir, dir_list, img_size, mesh_path, temporal_max, transform_img, transform_pose):
		super(SingleTCDataset, self).__init__()
		self.paired_samples = []
		for this_dir in dir_list:
			this_dir = os.path.join(base_dir, this_dir)
			num_files = 0
			for path in os.listdir(this_dir):
				if path.endswith(".txt"):
					num_files += 1
			candidate_buffer = set([i for i in range(temporal_max)])
			if num_files % 2 == 1:
				num_files -= 1
			for i in range(num_files):
				#print(i, candidate_buffer)
				if i in candidate_buffer:
					candidate_buffer.remove(i)
					partner = np.random.choice(list(candidate_buffer))
					self.paired_samples.append((this_dir, i, partner))
					candidate_buffer.remove(partner)
				if i + temporal_max < num_files:
					candidate_buffer.add(i + temporal_max)
		#print(self.paired_samples)
		self.transform_img, self.transform_pose = transform_img, transform_pose
		self.img_size = img_size
		if mesh_path is None:
			self.plotter = None
		else:
			self.plotter = pv.Plotter(off_screen = True, window_size = (img_size, img_size))
			self.plotter.add_mesh(pv.read(mesh_path))
	
	def __len__(self):
		return len(self.paired_samples)

	def __getitem__(self, idx):
		path, i1, i2 = self.paired_samples[idx]

		img_stack, pose_stack = [], []
		for idx in [i1, i2]:
			img, pose = load_img_pose(path, idx, self.plotter)
			if self.transform_img is not None:
				img = self.transform_img(img)
			if self.transform_pose is not None:
				pose = self.transform_pose(pose, True)
			img_stack.append(img)
			pose_stack.append(pose)

		return torch.stack(img_stack, axis = 0), torch.stack(pose_stack, axis = 0)
