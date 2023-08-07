import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import csv
from tqdm import tqdm
import cv2
from scipy.signal import savgol_filter

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

min_d, max_d = 1.2504748, 143.0871
clip_value = 120
window_size = 41
polyorder = 3
focal_length = 5
cl_idx = 0
cur_fold = 0
move_rate = 0.3
reset_step = 5
base_orientation = np.array([0, 0, 1])

mesh_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
cl_path = f"./CL/CL{cl_idx}.dat"
img_path = f"./depths-images/CL{cl_idx}-fold{cur_fold}"
rotation_path = "./relative_quats.npy"

points = []
radiuses = []
with open(cl_path, newline = "") as f:
	reader = csv.DictReader(f, delimiter = " ")
	for row in reader:
		points.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
		radiuses.append(float(row["MaximumInscribedSphereRadius"]))
points.reverse()
radiuses.reverse()
points = np.array(points)
radiuses = np.array(radiuses)
points = savgol_filter(points, window_size, polyorder, axis = 0)

relative_rotation_corpus = np.load(rotation_path)

positions, orientations = [], []
for i in range(len(points)):
	this_position = points[i] + np.maximum(np.random.randn(3) * move_rate, 1) * radiuses[i]
	if i > 0:
		if i == 1 or i % reset_step == 0:
			prev_orientation = (this_position - prev_position) / np.linalg.norm(this_position - prev_position)
		else:
			relative_rot = relative_rotation_corpus[np.random.randint(len(relative_rotation_corpus))]
			prev_orientation = R.from_quat(relative_rot).apply(prev_orientation)
		positions.append(prev_position)
		orientations.append(prev_orientation)
	prev_position = this_position

positions, orientations = np.array(positions), np.array(orientations)
#positions = savgol_filter(positions, window_size, polyorder, axis = 0)
#orientations = savgol_filter(orientations, window_size, polyorder, axis = 0)

surface = pv.read(mesh_path)
p = pv.Plotter(off_screen = True)
p.add_mesh(surface)
camera = pv.Camera()
os.makedirs(img_path, exist_ok = True)

for i in range(len(positions)):
	quaternion = R.align_vectors(base_orientation.reshape(1, 3), orientations[i].reshape(1, 3))[0].as_quat()
	camera.position = positions[i]
	camera.focal_point = focal_length * orientations[i] + camera.position
	p.camera = camera
	p.show(auto_close = False)
	img = -p.get_image_depth()
	img = np.minimum(img, clip_value)
	img = (img - min_d) / (clip_value - min_d) * 255
	cv2.imwrite(os.path.join(img_path, f"{i:06d}.png"), img)
	np.savetxt(os.path.join(img_path, f"{i:06d}.txt"), np.concatenate([positions[i], quaternion]).reshape(1, -1), fmt = "%.5f")
