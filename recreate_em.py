import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm
import cv2
import argparse

from utils.camera_motion import camera_params
from utils.cl_geometry import project_to_cl, load_all_cls

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--em-base-path", type = str, default = "./depth-images")
	parser.add_argument("--cl-base-path", type = str, default = "./CL")
	parser.add_argument("--em-idx", type = int, default = 0)
	parser.add_argument("--proj2cl", action = "store_true", default = False)
	parser.add_argument("--getdep", action = "store_true", default = False)
	parser.add_argument("--window-size", type = int, default = 224)
	return parser.parse_args()

def main():
	args = get_args()
	surface = pv.read(args.mesh_path)
	p = pv.Plotter(off_screen = True, window_size = (args.window_size, args.window_size))
	p.add_mesh(surface)
	camera = pv.Camera()
	img_path = os.path.join(args.em_base_path, f"EM-virtual-{args.em_idx}")
	em_path = os.path.join(args.em_base_path, f"EM-{args.em_idx}")
	os.makedirs(img_path, exist_ok = True)
	all_cls = load_all_cls(args.cl_base_path)

	for i in tqdm(range(len(os.listdir(em_path)) // 2)):
		position = np.loadtxt(os.path.join(em_path, f"{i:06d}.txt")).reshape(-1)
		translation = position[ : 3]
		quaternion = position[3 : ]
		if args.proj2cl:
			translation = project_to_cl(translation, all_cls)
		orientation = R.from_quat(quaternion).apply(camera_params["forward_direction"])
		up = R.from_quat(quaternion).apply(camera_params["up_direction"])
		orientation = orientation / np.linalg.norm(orientation)
		up = up / np.linalg.norm(up)
		camera.position = translation
		camera.focal_point = camera_params["focal_length"] * orientation + camera.position
		camera.view_angle = camera_params["view_angle"]
		camera.up = up
		p.camera = camera
		p.show(auto_close = False)
		"""
		img = -p.get_image_depth()
		img = np.minimum(img, clip_value)
		img = (img - min_d) / (clip_value - min_d) * 255
		"""
		#img = p.screenshot(None, return_img = True)
		p.screenshot(os.path.join(img_path, f"{i:06d}.png"))
		if args.getdep:
			dep_img = -p.get_image_depth()
			np.save(os.path.join(img_path, f"{i:06d}.npy"), dep_img)
		#cv2.imwrite(os.path.join(img_path, f"{i:06d}.png"), img)
		#np.savetxt(os.path.join(img_path, f"{i:06d}.txt"), np.concatenate([positions[i], quaternion]).reshape(1, -1), fmt = "%.5f")

if __name__ == '__main__':
	main()
