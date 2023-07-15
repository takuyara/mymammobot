import torch
from torch import nn
from models.model_utils import get_mlp

class MLPSelector(nn.Module):
	def __init__(self, in_channels, num_neurons, dropout, num_inputs, output_dim):
		super(MLPSelector, self).__init__()
		self.mlp, prev_features = get_mlp(in_channels, num_neurons, dropout)
		self.fc = nn.Linear(prev_features, output_dim * num_inputs)
		self.num_inputs = num_inputs
		self.output_dim = output_dim
	def forward(self, x, *v):
		weights = self.mlp(x)
		weights = self.fc(weights)
		v = torch.stack(v, dim = -1)
		weights = weights.view(weights.size(0), self.output_dim, self.num_inputs)
		weights = nn.Softmax(dim = -1)(weights)
		out = torch.sum(v * weights, dim = -1)
		return out
