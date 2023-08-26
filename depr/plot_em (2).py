import os
import argparse
import numpy as np
import pyvista as pv
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from utils.cl_geometry import project_to_cl, load_all_cls

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--em-base-path", type = str, default = "./depth-images")
	parser.add_argument("--cl-base-path", type = str, default = "./CL")
	parser.add_argument("--em-idx", type = int, default = 0)
	parser.add_argument("--plot-traj", action = "store_true", default = False)
	parser.add_argument("--plot-dir", action = "store_true", default = False)
	parser.add_argument("--proj2cl", action = "store_true", default = False)
	parser.add_argument("-l", "--l", type = int, default = 0)
	parser.add_argument("-r", "--r", type = int, default = 10000)
	return parser.parse_args()

def main():
	args = get_args()

	surface = pv.read(args.mesh_path)
	p = pv.Plotter()
	p.add_mesh(surface, opacity = 0.5)
	all_cls = load_all_cls(args.cl_base_path)
	points = []
	em_path = os.path.join(args.em_base_path, f"EM-{args.em_idx}")
	for i in tqdm(range(args.l, min(len(os.listdir(em_path)) // 2, args.r))):
		position = np.loadtxt(os.path.join(em_path, f"{i:06d}.txt")).reshape(-1)
		translation = position[ : 3]
		quaternion = position[3 : ]
		if args.proj2cl:
			translation = project_to_cl(translation, all_cls)
		points.append(translation)
		if args.plot_dir:
			orientation = R.from_quat(quaternion).apply(np.array([0, 0, 1]))
			orientation = orientation / np.linalg.norm(orientation)
			p.add_mesh(pv.Arrow(translation, orientation), color = "red")
			orientation = R.from_quat(quaternion).apply(np.array([0, 1, 0]))
			orientation = orientation / np.linalg.norm(orientation)
			p.add_mesh(pv.Arrow(translation, orientation), color = "blue")
			orientation = R.from_quat(quaternion).apply(np.array([1, 0, 0]))
			orientation = orientation / np.linalg.norm(orientation)
			p.add_mesh(pv.Arrow(translation, orientation), color = "green")

	if args.plot_traj:
		poly = pv.lines_from_points(np.array(points))
		tube = poly.tube(radius = 0.05)
		p.add_mesh(tube, color = "black")

	p.show()

if __name__ == '__main__':
	main()
