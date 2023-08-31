import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
#from kornia.enhance import sharpness, adjust_contrast
#from kornia.filters import gaussian_blur2d
import torchvision.transforms.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split

class SimpleDataset(Dataset):
	def __init__(self, paired_paths):
		super(SimpleDataset, self).__init__()
		self.paired_paths = paired_paths

	def __len__(self):
		return len(self.paired_paths)

	def __getitem__(self, idx):
		r_p, v_p = self.paired_paths[idx]
		r_p, v_p = np.load(r_p), np.load(v_p)
		return r_p, v_p

def get_simple_loaders(paired_paths, test_size = 0.25, batch_size = 16, shuffle = True, num_workers = 0):
	train_paths, test_paths = train_test_split(paired_paths, test_size = test_size, random_state = 42, shuffle = True)
	train_dset, test_dset = SimpleDataset(train_paths), SimpleDataset(test_paths)
	train_dloader = DataLoader(train_dset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
	test_dloader = DataLoader(test_dset, batch_size = batch_size, shuffle = False, num_workers = num_workers)
	return train_dloader, test_dloader

class SFS2Mesh(nn.Module):
	def __init__(self, sfs_min, sfs_max, mesh_min, mesh_max, sharpness_factor = 20):
		super(SFS2Mesh, self).__init__()
		self.sfs_min, self.sfs_max = sfs_min, sfs_max
		self.mesh_min, self.mesh_max = mesh_min, mesh_max
		self._w = nn.Parameter(torch.tensor(1.))
		self._b = nn.Parameter(torch.tensor(0.))
		self.sharpness_factor = sharpness_factor

	def forward(self, sfs_img, mesh_img):
		initial_shape = sfs_img.shape
		sfs_img, mesh_img = sfs_img.unsqueeze(1), mesh_img.unsqueeze(1)
		#sfs_img = (sfs_img - self.sfs_min) / (self.sfs_max - self.sfs_min)
		#sfs_img = F.adjust_sharpness(sfs_img, self.sharpness_factor)
		#mesh_img = (mesh_img - self.mesh_min) / (self.mesh_max - self.mesh_min)
		#sfs_img = adjust_contrast(sfs_img, self.contrast_factor)
		#sfs_img = sharpness(sfs_img, self.sharpness_factor)
		#sfs_img = gaussian_blur2d(sfs_img, (self.gaussian_kernel, self.gaussian_kernel), (self.gaussian_sigma, self.gaussian_sigma))
		sfs_img_f, mesh_img_f = sfs_img.reshape(sfs_img.size(0), -1), mesh_img.reshape(mesh_img.size(0), -1)
		sfs_solved_f = self._w * sfs_img_f + self._b
		loss = nn.MSELoss()(sfs_solved_f, mesh_img_f)
		return loss, sfs_solved_f.reshape(*initial_shape)


class Mesh2SFS(nn.Module):
	def __init__(self, sfs_min, sfs_max, mesh_min, mesh_max, clip = 50, blur_kernel = 13, gamma = 1):
		super(Mesh2SFS, self).__init__()
		self.sfs_min, self.sfs_max = sfs_min, sfs_max
		self.mesh_min, self.mesh_max = mesh_min, mesh_max
		#self._w = nn.Parameter(torch.tensor(1.))
		self._w = nn.Parameter(torch.tensor(0.13236457109451294))
		#self._b = nn.Parameter(torch.tensor(0.))
		self._b = nn.Parameter(torch.tensor(2.5300467014312744))
		self.blur_kernel = blur_kernel
		self.gamma = gamma
		self.clip = clip

	def forward(self, sfs_img, mesh_img):
		initial_shape = sfs_img.shape
		sfs_img, mesh_img = sfs_img.unsqueeze(1), mesh_img.unsqueeze(1)
		mesh_img = transforms.GaussianBlur(21, 7)(mesh_img)
		#mesh_img = (mesh_img - self.mesh_min) / (self.mesh_max - self.mesh_min)
		#mesh_img = (mesh_img - self.mesh_min) / (self.clip - self.mesh_min)
		#mesh_img = torch.clamp(mesh_img, 0, 1)
		#mesh_img = F.adjust_gamma(mesh_img, gamma = self.gamma)
		#mesh_img = F.gaussian_blur(mesh_img, kernel_size = self.blur_kernel)
		sfs_img_f, mesh_img_f = sfs_img.reshape(sfs_img.size(0), -1), mesh_img.reshape(mesh_img.size(0), -1)
		mesh_solved_f = self._w * mesh_img_f + self._b
		loss = nn.MSELoss()(sfs_img_f, mesh_solved_f)
		return loss, mesh_solved_f.reshape(*initial_shape)
