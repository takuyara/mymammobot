import torch
from torch import nn
from models.model_utils import get_mlp

class MLPSelector(nn.Module):
	def __init__(self, in_channels, num_neurons, dropout, num_inputs):
		super(MLPSelector, self).__init__()
		self.mlp, prev_features = get_mlp(in_channels, num_neurons, dropout)
		self.fc = nn.Sequential(nn.Linear(prev_features, num_inputs), nn.Softmax(dim = -1))
	def forward(self, x, *v):
		weights = self.mlp(x)
		weights = self.fc(weights)
		v = torch.stack(v, dim = -1)
		out = torch.sum(v * weights, dim = -1)
		return out
