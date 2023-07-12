import os
import numpy as np
from scipy.spatial.transform import Rotation

dir_list = [
	"E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\EM",
	"E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_17-08-08_Phantom_2\\EM",
	"E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_17-12-48_Phantom_3\\EM",
]

def read_files(dir_list):
	velocity_norms, relative_quats = [], []
	for this_dir in dir_list:
		for i in range(len(os.listdir(this_dir))):
			with open(os.path.join(this_dir, f"{i}.txt")) as f:
				this_num = np.array([float(x) for x in f.read().split()])
			if i > 0:
				this_norm = np.linalg.norm(this_num[ : 3] - prev_num[ : 3])
				#this_quat = Rotation.align_vectors(this_num[3 : ], prev_num[3 : ])[0]
				this_quat = Rotation.from_quat(this_num[3 : ]) * Rotation.from_quat(prev_num[3 : ]).inv()
				velocity_norms.append(this_norm)
				relative_quats.append(this_quat.as_quat())
			prev_num = this_num
	return np.array(velocity_norms), np.array(relative_quats)

velocity_norms, relative_quats = read_files(dir_list)
print(velocity_norms.shape, relative_quats.shape)
np.save("velocity_norms.npy", velocity_norms)
np.save("relative_quats.npy", relative_quats)
print(relative_quats, np.mean(relative_quats, axis = 0))
