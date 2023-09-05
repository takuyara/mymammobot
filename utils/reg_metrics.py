import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score, accuracy_score

from utils.pose_utils import quat_angular_error, revert_quat

class TransL2Loss(nn.Module):
	def __init__(self):
		super(TransL2Loss, self).__init__()
		self.base_loss = nn.MSELoss()
	def forward(self, inp, tgt):
		return self.base_loss(inp[..., : 3], tgt[..., : 3])

class BalancedL1Loss(nn.Module):
	def __init__(self):
		super(BalancedL1Loss, self).__init__()
		self._beta = nn.Parameter(torch.tensor(0.))
		self._gamma = nn.Parameter(torch.tensor(-3.))
		self.base_loss = nn.L1Loss()

	def forward(self, inputs, targets):
		trans_loss = self.base_loss(inputs[..., : 3], targets[..., : 3])
		rot_loss = self.base_loss(inputs[..., 3 : ], targets[..., 3 : ])
		loss = trans_loss * torch.exp(-self._beta) + rot_loss * torch.exp(-self._gamma) + self._beta + self._gamma
		return loss

class TCLoss(nn.Module):
	def __init__(self, base_loss, relative_coef = 1.):
		super(TCLoss, self).__init__()
		self.base_loss = base_loss
		self.relative_coef = relative_coef

	def forward(self, inputs, targets):
		loss = self.base_loss(inputs, targets)
		if inputs.dim == 3:
			inputs_rela = inputs[ : , 0, ...] - inputs[ : , 1, ...]
			targets_rela = targets[ : , 0, ...] - targets[ : , 1, ...]
			loss += self.relative_coef * self.base_loss(inputs_rela, targets_rela)
		return loss

class CombineLoss(nn.Module):
	def __init__(self, balance_rate = 1.):
		super(CombineLoss, self).__init__()
		self.cls_loss = nn.CrossEntropyLoss()
		self.reg_loss = nn.MSELoss()
		self.balance_rate = balance_rate
	def forward(self, inputs, targets):
		cls_input, reg_input = inputs
		cls_target, reg_target = targets
		return self.cls_loss(cls_input, cls_target) + self.balance_rate * self.reg_loss(reg_input, reg_target)

class Metrics_Cls:
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
		self.inp_lbs = []
		self.tgt_lbs = []
		self.inp_regs = []
		self.tgt_regs = []
	def new_copy(self):
		return Metrics_Cls(self.loss_fun, self.inv_trans, self.main_metric_name, self.rot_coef)
	def main_metric(self):
		inp = np.concatenate(self.inp_lbs, axis = 0)
		tgt = np.concatenate(self.tgt_lbs, axis = 0)
		return -f1_score(tgt, inp, average = "micro")
		"""
		if self.main_metric_name == "loss":
			return self.loss()
		elif self.main_metric_name == "trans_err":
			return np.mean(self.trans_errors)
		elif self.main_metric_name == "rot_err":
			return np.mean(self.rot_errors)
		elif self.main_metric_name == "comb_err":
			return np.mean(self.trans_errors) + self.rot_coef * np.mean(self.rot_errors)
		"""
	def add_batch(self, inp, tgt):
		self.sum_loss += self.loss_fun(inp, tgt)
		inp_cls, inp_reg = inp
		tgt_cls, tgt_reg = tgt
		self.n_samples += inp_cls.size(0)
		inp_cls, tgt_cls = inp_cls.cpu().detach().numpy(), tgt_cls.cpu().detach().numpy()
		inp_reg, tgt_reg = inp_reg.cpu().detach().numpy(), tgt_reg.cpu().detach().numpy()
		#inp, tgt = inp.reshape(-1, inp.shape[-1]), tgt.reshape(-1, tgt.shape[-1])
		#print("Before: ", inp.shape, tgt.shape)
		inp_cls, tgt_cls = self.inv_trans(inp_cls), self.inv_trans(tgt_cls)
		#print("After: ", inp.shape, tgt.shape)
		self.inp_lbs.append(inp_cls)
		self.tgt_lbs.append(tgt_cls)
		self.inp_regs.append(inp_reg)
		self.tgt_regs.append(tgt_reg)
		"""
		for i in range(len(inp)):
			self.trans_errors.append(np.linalg.norm(inp[i, : 3] - tgt[i, : 3]))
			self.rot_errors.append(quat_angular_error(revert_quat(inp[i, 3 : ]), revert_quat(tgt[i, 3 : ])))
		"""
	def loss(self):
		return self.sum_loss / self.n_samples
	def get_dict(self):
		"""
		return {
			"loss": self.sum_loss / self.n_samples, "translation_error": np.mean(self.trans_errors), "translation_error_std": np.std(self.trans_errors),
			"rotation_error": np.mean(self.rot_errors), "rotation_error_std": np.std(self.rot_errors), "main_metric": self.main_metric(),
		}
		"""
		inp = np.concatenate(self.inp_lbs, axis = 0)
		tgt = np.concatenate(self.tgt_lbs, axis = 0)
		inp_r = np.concatenate(self.inp_regs, axis = 0)
		tgt_r = np.concatenate(self.tgt_regs, axis = 0)
		return {
			"loss": self.sum_loss / self.n_samples, "f1_macro": f1_score(tgt, inp, average = "macro"), "acc": accuracy_score(tgt, inp), "reg_mae": np.mean(np.abs(inp_r - tgt_r)),
		}
	def __repr__(self):
		dct = self.get_dict()
		"""
		repr_str = "loss: {:.5f}; trans_error: {:.2f}, {:.2f}; rot_error: {:.2f}, {:.2f}".format(dct["loss"], dct["translation_error"], dct["translation_error_std"], dct["rotation_error"], dct["rotation_error_std"])
		if isinstance(self.loss_fun, BalancedL1Loss):
			repr_str = repr_str + f" beta: {self.loss_fun._beta.item():.4f}, gamma: {self.loss_fun._gamma.item():.4f}"
		"""
		repr_str = "loss: {:.5f}; f1_macro: {:.4f}; acc: {:.4f}; reg_mae: {:.4f}".format(dct["loss"], dct["f1_macro"], dct["acc"], dct["reg_mae"])
		return repr_str


class Metrics_Reg:
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
		return Metrics_Reg(self.loss_fun, self.inv_trans, self.main_metric_name, self.rot_coef)
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
		inp, tgt = inp.reshape(-1, inp.shape[-1]), tgt.reshape(-1, tgt.shape[-1])
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
