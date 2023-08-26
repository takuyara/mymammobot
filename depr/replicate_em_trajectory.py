import os
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

focal_length = 300

mesh_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
em_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\EM"

surface = pv.read(mesh_path)

for i in range(len(os.listdir(em_path))):
	p = pv.Plotter()
	p.add_mesh(surface)
	camera = pv.Camera()
	general_pose = np.loadtxt(os.path.join(em_path, f"{i}.txt")).reshape(-1)
	position = general_pose[ : 3]
	orientation = R.from_quat(general_pose[3 : ]).as_matrix().T @ np.array([[0, 0, 1]]).T
	#position = -R.from_quat(general_pose[3 : ]).as_matrix().T @ position.reshape(3, 1)
	position, orientation = position.reshape(3), orientation.reshape(3)
	camera.position = position
	camera.focal_point = focal_length * orientation + position
	p.camera = camera
	p.show(auto_close = False)
