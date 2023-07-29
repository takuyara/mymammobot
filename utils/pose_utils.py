import numpy as np
from torch import nn
from scipy.spatial.transform import Rotation as R

def get_3dof_quat(quat):
	quat *= np.sign(quat[0])
	if np.linalg.norm(quat[1 : ]) != 0:
		quat_3dof = np.arccos(quat[0]) * quat[1 : ] / np.linalg.norm(quat[1 : ])
	else:
		quat_3dof = np.zeros(3)
	return quat_3dof

def revert_quat(quat_3dof):
	norm = np.linalg.norm(quat_3dof)
	quat = np.concatenate([[np.cos(norm)], np.sinc(norm / np.pi) * quat_3dof])
	return quat

def get_6dof_pose_label(pose):
	pose = pose.reshape(-1)
	trans, quat = pose[ : 3], pose[3 : ]
	return np.concatenate([trans, get_3dof_quat(quat)])

def quat_angular_error(q1, q2):
	d = np.abs(np.dot(q1, q2))
	d = min(1, max(-1, d))
	dlt = 2 * np.arccos(d) * 180 / np.pi
	return dlt

def compute_rotation_quaternion(src, tgt):
	src, tgt = src / np.linalg.norm(src), tgt / np.linalg.norm(tgt)
	crs = np.cross(src, tgt)
	if np.linalg.norm(crs) > 0:
		dot = np.dot(src, tgt)
		K = np.array([[0, -crs[2], crs[1]], [crs[2], 0, -crs[0]], [-crs[1], crs[0], 0]])
		K = np.eye(3) + K + K.dot(K) * ((1 - dot) / (np.linalg.norm(crs) ** 2))
	else:
		K = np.eye(3)
	return R.from_matrix(K).as_quat()

class Metrics:
	def __init__(self, loss_fun, inv_trans, main_metric, rot_coef = None):
		assert main_metric in ["loss", "trans_err", "rot_err", "comb_err"]
		self.inv_trans = inv_trans
		self.main_metric_name = main_metric
		self.rot_coef = rot_coef
		self.n_samples = 0
		self.sum_loss = 0
		self.loss_fun = loss_fun
		self.sum_trans_error = 0
		self.sum_rot_error = 0
	def new_copy(self):
		return Metrics(self.loss_fun, self.inv_trans, self.main_metric_name, self.rot_coef)
	def main_metric(self):
		if self.main_metric_name == "loss":
			return self.loss()
		elif self.main_metric_name == "trans_err":
			return self.sum_trans_error / self.n_samples
		elif self.main_metric_name == "rot_err":
			return self.sum_rot_error / self.n_samples
		elif self.main_metric_name == "comb_err":
			return (self.sum_trans_error + self.rot_coef * self.sum_rot_error) / self.n_samples
	def add_batch(self, inp, tgt):
		self.n_samples += inp.size(0)
		self.sum_loss += self.loss_fun(inp, tgt)
		inp, tgt = inp.cpu().detach().numpy(), tgt.cpu().detach().numpy()
		inp, tgt = self.inv_trans(inp), self.inv_trans(tgt)
		for i in range(len(inp)):
			self.sum_trans_error += np.linalg.norm(inp[i, : 3] - tgt[i, : 3])
			self.sum_rot_error += quat_angular_error(revert_quat(inp[i, 3 : ]), revert_quat(tgt[i, 3 : ]))
	def loss(self):
		return self.sum_loss / self.n_samples
	def get_dict(self):
		return {"loss": self.sum_loss / self.n_samples, "translation_error": self.sum_trans_error / self.n_samples, "rotation_error": self.sum_rot_error / self.n_samples, "main_metric": self.main_metric()}
	def __repr__(self):
		return ",".join([f"{m_name}: {m_value:.5f}" for m_name, m_value in self.get_dict().items()])
