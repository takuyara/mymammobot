import torch
from torch import nn

class CustomBatchNorm(nn.Module):
	def __init__(self, num_features, dim = 1, momenteum = 0.1, eps = 1e-7):
		super(CustomBatchNorm, self).__init__()
		if dim == 1:
			self.train_bn = nn.BatchNorm1d(num_features)
			self.dims = (0,)
		elif dim == 2:
			self.train_bn = nn.BatchNorm2d(num_features)
			self.dims = (0, 2, 3)
		else:
			raise NotImplementedError
		self.running_mean_eval = torch.zeros(num_features)
		self.running_var_eval = torch.ones(num_features)
		self.eps = eps
		self.momenteum = momenteum
		self.was_training = False
	def forward(self, x):
		if self.training:
			x = self.train_bn(x)
			self.was_training = True
		else:
			if self.was_training:
				self.was_training = False
				self.running_mean_eval = self.train_bn.running_mean
				self.running_var_eval = self.train_bn.running_var
			self.running_mean_eval = self.running_mean_eval.to(x.device)
			self.running_var_eval = self.running_var_eval.to(x.device)
			self.running_mean_eval = self.running_mean_eval * (1 - self.momenteum) + torch.mean(x, dim = self.dims) * self.momenteum
			self.running_var_eval = self.running_var_eval * (1 - self.momenteum) + torch.var(x, dim = self.dims) * self.momenteum
			x = (x - self.running_mean_eval) / ((self.running_var_eval + self.eps) ** 0.5)
		return x
