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
