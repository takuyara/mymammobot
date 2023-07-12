import numpy as np
def get_6dof_pose_label(pose):
	pose = pose.reshape(-1)
	trans, quat = pose[ : 3], pose[3 : ]
	quat *= np.sign(quat[0])
	if np.linalg.norm(quat[1 : ]) != 0:
		quat_3dof = np.arccos(quat[0]) * quat[1 : ] / np.linalg.norm(quat[1 : ])
	else:
		quat_3dof = np.zeros(3)
	return np.concatenate([trans, quat_3dof])
