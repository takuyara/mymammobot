import torch
from torch import nn
from models.model_utils import get_mlp

class MLPFusePredictor(nn.Module):
	def __init__(self, f1_dim, f2_dim, b1_neurons, b2_neurons, fn_neurons, output_dim, dropout, fuse_mode):
		super(MLPFusePredictor, self).__init__()
		self.fuse_mode = fuse_mode
		self.mlp_b1, out_b1 = get_mlp(f1_dim, b1_neurons, dropout)
		self.mlp_b2, out_b2 = get_mlp(f2_dim, b2_neurons, dropout)
		if fuse_mode == "cat":
			out_dim = out_b1 + out_b2
		elif fuse_mode == "plus":
			assert out_b1 == out_b2
			out_dim = out_b1
		else:
			raise NotImplementedError
		self.mlp_fn, out_fn = get_mlp(out_dim, fn_neurons, dropout)
		self.fc = nn.Linear(out_fn, output_dim)
	def forward(self, x1, x2):
		x1 = self.mlp_b1(x1)
		x2 = self.mlp_b2(x2)
		if self.fuse_mode == "cat":
			xc = torch.cat((x1, x2), dim = -1)
		elif self.fuse_mode == "plus":
			xc = x1 + x2
		xc = self.mlp_fn(xc)
		xc = self.fc(xc)
		return xc
