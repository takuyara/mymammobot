import numpy as np
from torch import nn

def get_6dof_pose_label(pose):
	pose = pose.reshape(-1)
	trans, quat = pose[ : 3], pose[3 : ]
	quat *= np.sign(quat[0])
	if np.linalg.norm(quat[1 : ]) != 0:
		quat_3dof = np.arccos(quat[0]) * quat[1 : ] / np.linalg.norm(quat[1 : ])
	else:
		quat_3dof = np.zeros(3)
	return np.concatenate([trans, quat_3dof])

def revert_quat(quat_3dof):
	norm = np.linalg.norm(quat_3dof)
	quat = np.concatenate([[np.cos(norm)], np.sinc(norm / np.pi) * quat_3dof])
	return quat

def quat_angular_error(q1, q2):
	d = np.abs(np.dot(q1, q2))
	d = min(1, max(-1, d))
	dlt = 2 * np.arccos(d) * 180 / np.pi
	return dlt

class Metrics:
	def __init__(self, inv_trans, loss_fun = nn.MSELoss()):
		self.inv_trans = inv_trans
		self.n_samples = 0
		self.sum_loss = 0
		self.loss_fun = loss_fun
		self.sum_trans_error = 0
		self.sum_rot_error = 0
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
		return {"loss": self.sum_loss / self.n_samples, "translation_error": self.sum_trans_error / self.n_samples, "rotation_error": self.sum_rot_error / self.n_samples}
	def __repr__(self):
		return ",".join([f"{m_name}: {m_value:.5f}" for m_name, m_value in self.get_dict().items()])
