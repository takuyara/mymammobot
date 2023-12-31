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
from utils.pose_utils import compute_rotation_quaternion
from utils.cl_geometry import load_cl

import warnings
warnings.filterwarnings("ignore", category = UserWarning)


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
cl_idx = int(sys.argv[1])
cur_fold = int(sys.argv[2])
move_rate = 0.3
reset_step = 5
rot_euler_angle = 5
clipping_range = (1, 200)
base_orientation = np.array([1, 1, 1])

mesh_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
cl_base_path = "./CL"
img_path = f"./depth-images/CL{cl_idx}-fold{cur_fold}"
rotation_path = "./relative_quats.npy"

points, radiuses = load_cl(cl_base_path, cl_idx)
points = savgol_filter(points, window_size, polyorder, axis = 0)

relative_rotation_corpus = np.load(rotation_path)

positions, orientations = [], []
for i in range(len(points)):
	this_position = points[i] + np.minimum(np.random.randn(3) * move_rate, 1) * radiuses[i]
	if i > 0:
		noise_rot = R.from_euler("xyz", np.random.rand(3) * rot_euler_angle * 2 - rot_euler_angle, degrees = True)
		prev_orientation = noise_rot.apply(points[i] - points[i - 1])
		prev_orientation = prev_orientation / np.linalg.norm(prev_orientation)
		positions.append(prev_position)
		orientations.append(prev_orientation)
	prev_position = this_position


	#this_position = points[i]
	"""
	if i > 0:
		prev_orientation = (this_position - prev_position) / np.linalg.norm(this_position - prev_position)
		if i == 1 or i % reset_step == 0:
			prev_orientation = (this_position - prev_position) / np.linalg.norm(this_position - prev_position)
		else:
			relative_rot = relative_rotation_corpus[np.random.randint(len(relative_rotation_corpus))]
			prev_orientation = R.from_quat(relative_rot).apply(prev_orientation)
		positions.append(prev_position)
		orientations.append(prev_orientation)
	prev_position = this_position
	"""

positions, orientations = np.array(positions), np.array(orientations)
positions = savgol_filter(positions, window_size, polyorder, axis = 0)
orientations = savgol_filter(orientations, window_size, polyorder, axis = 0)

surface = pv.read(mesh_path)
p = pv.Plotter(off_screen = True)
p.add_mesh(surface)
camera = pv.Camera()
camera.clipping_range = clipping_range
os.makedirs(img_path, exist_ok = True)

for i in range(len(positions)):
	#quaternion = R.align_vectors(base_orientation.reshape(1, 3), orientations[i].reshape(1, 3))[0].as_quat()
	#quaternion = R.from_matrix(rotation_matrix_from_vectors(base_orientation, orientations[i])).as_quat()
	quaternion = compute_rotation_quaternion(base_orientation, orientations[i])
	#print(orientations[i], quaternion)
	camera.position = positions[i]
	camera.focal_point = focal_length * orientations[i] + camera.position
	camera.view_angle = view_angle
	p.camera = camera
	p.show(auto_close = False)
	img = -p.get_image_depth()
	img = np.minimum(img, clip_value)
	img = (img - min_d) / (clip_value - min_d) * 255
	cv2.imwrite(os.path.join(img_path, f"{i:06d}.png"), img)
	np.savetxt(os.path.join(img_path, f"{i:06d}.txt"), np.concatenate([positions[i], quaternion]).reshape(1, -1), fmt = "%.5f")
