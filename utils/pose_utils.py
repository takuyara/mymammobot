import numpy as np
from torch import nn
from scipy.spatial.transform import Rotation as R

from ds_gen.camera_features import camera_params

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

def camera_pose_to_train_pose(position, orientation, up, reshape_for_output = False):
	# From camera coordinate to global coordinate
	cam_coord = np.stack([camera_params["forward_direction"], camera_params["up_direction"]], axis = 0)
	global_coord = np.stack([orientation, up], axis = 0)
	"""
	print(np.linalg.norm(orientation), np.linalg.norm(up))
	print(cam_coord.shape, global_coord.shape)
	"""
	rot = R.align_vectors(global_coord, cam_coord)[0]
	"""
	print(np.allclose(rot.apply(camera_params["forward_direction"]), orientation))
	print(np.allclose(rot.apply(camera_params["up_direction"]), up))

	print(rot.apply(camera_params["forward_direction"]), orientation)
	print(rot.apply(camera_params["up_direction"]), up)
	"""
	res = np.concatenate([position, rot.as_quat()], axis = 0)
	#print(res.shape)
	if reshape_for_output:
		res = res.reshape(1, -1)
	return res

class Metrics:
	def __init__(self, loss_fun, inv_trans, main_metric, rot_coef = None):
		assert main_metric in ["loss", "trans_err", "rot_err", "comb_err"]
		self.inv_trans = inv_trans
		self.main_metric_name = main_metric
		self.rot_coef = rot_coef
		self.n_samples = 0
		self.sum_loss = 0
		self.loss_fun = loss_fun
		self.trans_errors = 0
		self.rot_errors = 0
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
		return "loss: {:.5f}; trans_error: {:.2f}, {:.2f}; rot_error: {:.2f}, {:.2f}".format(dct["loss"], dct["translation_error"], dct["translation_error_std"], dct["rotation_error"], dct["rotation_error_std"])
