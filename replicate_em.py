import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import csv
from tqdm import tqdm
import cv2
from scipy.signal import savgol_filter
import sys

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

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

"""
def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions
"""

min_d, max_d = 1.2504748, 143.0871
clip_value = 120
window_size = 41
polyorder = 3
focal_length = 214
view_angle = 120
cl_idx = 0
move_rate = 0.3
reset_step = 5
rot_euler_angle = 5
clipping_range = (1, 200)
base_orientation = np.array([1, 1, 1])

mesh_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
em_path = f"./depth-images/EM-{cl_idx}"
img_path = f"./depth-images/EM-{cl_idx}-replicate"

surface = pv.read(mesh_path)
p = pv.Plotter(off_screen = True)
p.add_mesh(surface)
camera = pv.Camera()
camera.clipping_range = clipping_range
os.makedirs(img_path, exist_ok = True)


for i in range(len(os.listdir(em_path)) // 2):
	position = np.loadtxt(os.path.join(em_path, f"{i:06d}.txt")).reshape(-1)
	translation = position[ : 3]
	quaternion = position[3 : ]
	orientation = R.from_quat(quaternion).apply(np.array([1, 0, 0]))
	orientation = orientation / np.linalg.norm(orientation)
	camera.position = translation
	camera.focal_point = focal_length * orientation + camera.position
	camera.view_angle = view_angle
	p.camera = camera
	p.show(auto_close = False)
	img = -p.get_image_depth()
	img = np.minimum(img, clip_value)
	img = (img - min_d) / (clip_value - min_d) * 255
	cv2.imwrite(os.path.join(img_path, f"{i:06d}.png"), img)
	#np.savetxt(os.path.join(img_path, f"{i:06d}.txt"), np.concatenate([positions[i], quaternion]).reshape(1, -1), fmt = "%.5f")
