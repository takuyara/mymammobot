import os
import argparse
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt

from utils.stats import fit_best_distrib, get_pdf_curve, get_num_bins

def direction_velocity_angle():
	em_base_path = "./depth-images/EM-{}"
	angles = []
	for cl_idx in [0]:
		em_path = em_base_path.format(cl_idx)
		all_positions, all_orientations = [], []
		for i in range(len(os.listdir(em_path)) // 2):
			camera_pose = np.loadtxt(os.path.join(em_path, f"{i:06d}.txt")).reshape(-1)
			translation = camera_pose[ : 3]
			quaternion = camera_pose[3 : ]
			orientation = R.from_quat(quaternion).apply(np.array([0, 0, 1]))
			orientation = orientation / np.linalg.norm(orientation)
			all_positions.append(translation)
			all_orientations.append(orientation)
		all_positions, all_orientations = np.stack(all_positions, axis = 0), np.stack(all_orientations, axis = 0)
		velocities = all_positions[1 : ] - all_positions[ : -1]
		for i in range(len(velocities)):
			if np.abs(np.linalg.norm(velocities[i])) < 1e-5:
				continue
			velocity_unit = velocities[i] / np.linalg.norm(velocities[i])
			this_angle = np.arccos(np.clip(np.dot(all_orientations[i], velocity_unit), -1.0, 1.0))
			if i < 60:
				print(np.rad2deg(this_angle))
			angles.append(this_angle)
	print(np.rad2deg(circmean(angles)), np.rad2deg(circstd(angles)))
	"""
	plt.hist(np.rad2deg(angles))
	plt.show()
	"""

def velocity_norm():
	em_base_path = "./depth-images/EM-{}"
	velocity_norms = []
	for cl_idx in [0, 1, 2]:
		em_path = em_base_path.format(cl_idx)
		all_positions, all_orientations = [], []
		for i in range(len(os.listdir(em_path)) // 2):
			camera_pose = np.loadtxt(os.path.join(em_path, f"{i:06d}.txt")).reshape(-1)
			translation = camera_pose[ : 3]
			quaternion = camera_pose[3 : ]
			orientation = R.from_quat(quaternion).apply(np.array([0, 0, 1]))
			orientation = orientation / np.linalg.norm(orientation)
			all_positions.append(translation)
			all_orientations.append(orientation)
		all_positions, all_orientations = np.stack(all_positions, axis = 0), np.stack(all_orientations, axis = 0)
		velocities = all_positions[1 : ] - all_positions[ : -1]
		for i in range(len(velocities)):
			velocity_norms.append(np.linalg.norm(velocities[i]))
	print(np.mean(velocity_norms), np.std(velocity_norms), np.max(velocity_norms))
	velocity_norms = np.clip(velocity_norms, 0, 6)

	sse, distrib_name, distrib, params = fit_best_distrib(velocity_norms)
	print(sse, distrib_name, params)
	
	plt.hist(velocity_norms, bins = get_num_bins(velocity_norms), density = True)
	plt.plot(*get_pdf_curve(distrib, params), "r-", lw = 5, alpha = 0.6)
	plt.show()


def main():
	#direction_velocity_angle()
	velocity_norm()
	"""
	loc = -0.0604376084
	scale = 0.447411730651
	c = 1.10161126717047
	from scipy.stats import fatiguelife
	x = np.linspace(fatiguelife.ppf(0.01, c, loc = loc, scale = scale), fatiguelife.ppf(0.99, c, loc = loc, scale = scale), 100)
	plt.plot(x, fatiguelife.pdf(x, c, loc = loc, scale = scale), "r-", lw = 5, alpha = 0.6)
	plt.show()
	"""

if __name__ == '__main__':
	main()
