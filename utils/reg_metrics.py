import numpy as np
import torch
from torch import nn

from utils.pose_utils import quat_angular_error, revert_quat

class BalancedL1Loss(nn.Module):
	def __init__(self):
		super(BalancedL1Loss, self).__init__()
		self._beta = nn.Parameter(torch.tensor(0.))
		self._gamma = nn.Parameter(torch.tensor(0.))
		self.base_loss = nn.L1Loss()

	def forward(self, inputs, targets):
		trans_loss = self.base_loss(inputs[..., : 3], targets[..., : 3])
		rot_loss = self.base_loss(inputs[..., 3 : ], targets[..., 3 : ])
		loss = trans_loss * torch.exp(-self._beta) + rot_loss * torch.exp(-self._gamma) + self._beta + self._gamma
		return loss

class Metrics:
	def __init__(self, loss_fun, inv_trans, main_metric, rot_coef = None):
		assert main_metric in ["loss", "trans_err", "rot_err", "comb_err"]
		self.inv_trans = inv_trans
		self.main_metric_name = main_metric
		self.rot_coef = rot_coef
		self.n_samples = 0
		self.sum_loss = 0
		self.loss_fun = loss_fun
		self.trans_errors = []
		self.rot_errors = []
	def new_copy(self):
		return Metrics(self.loss_fun, self.inv_trans, self.main_metric_name, self.rot_coef)
	def main_metric(self):
		if self.main_metric_name == "loss":
			return self.loss()
		elif self.main_metric_name == "trans_err":
			return np.mean(self.trans_errors)
		elif self.main_metric_name == "rot_err":
			return np.mean(self.rot_errors)
		elif self.main_metric_name == "comb_err":
			return np.mean(self.trans_errors) + self.rot_coef * np.mean(self.rot_errors)
	def add_batch(self, inp, tgt):
		self.n_samples += inp.size(0)
		self.sum_loss += self.loss_fun(inp, tgt)
		inp, tgt = inp.cpu().detach().numpy(), tgt.cpu().detach().numpy()
		inp, tgt = self.inv_trans(inp), self.inv_trans(tgt)
		for i in range(len(inp)):
			self.trans_errors.append(np.linalg.norm(inp[i, : 3] - tgt[i, : 3]))
			self.rot_errors.append(quat_angular_error(revert_quat(inp[i, 3 : ]), revert_quat(tgt[i, 3 : ])))
	def loss(self):
		return self.sum_loss / self.n_samples
	def get_dict(self):
		return {
			"loss": self.sum_loss / self.n_samples, "translation_error": np.mean(self.trans_errors), "translation_error_std": np.std(self.trans_errors),
			"rotation_error": np.mean(self.rot_errors), "rotation_error_std": np.std(self.rot_errors), "main_metric": self.main_metric(),
		}
	def __repr__(self):
		dct = self.get_dict()
		repr_str = "loss: {:.5f}; trans_error: {:.2f}, {:.2f}; rot_error: {:.2f}, {:.2f}".format(dct["loss"], dct["translation_error"], dct["translation_error_std"], dct["rotation_error"], dct["rotation_error_std"])
		if isinstance(self.loss_fun, BalancedL1Loss):
			repr_str = repr_str + f" beta: {self.loss_fun._beta.item():.4f}, gamma: {self.loss_fun._gamma.item():.4f}"
		return repr_str
