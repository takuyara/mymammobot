import torch
from torch import nn
from models.model_utils import get_mlp

class MLPFusePredictor(nn.Module):
	def __init__(self, f1_dim, f2_dim, b1_neurons, b2_neurons, fn_neurons, output_dim, dropout):
		super(MLPFusePredictor, self).__init__()
		self.mlp_b1, out_b1 = get_mlp(f1_dim, b1_neurons, dropout)
		self.mlp_b2, out_b2 = get_mlp(f2_dim, b2_neurons, dropout)
		self.mlp_fn, out_fn = get_mlp(out_b1 + out_b2, fn_neurons, dropout)
		self.fc = nn.Linear(out_fn, output_dim)
	def forward(self, x1, x2):
		x1 = self.mlp_b1(x1)
		x2 = self.mlp_b2(x2)
		xc = torch.cat((x1, x2), dim = -1)
		xc = self.mlp_fn(xc)
		xc = self.fc(xc)
		return xc
