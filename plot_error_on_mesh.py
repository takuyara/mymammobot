import os
import csv
import numpy as np
import pyvista as pv

airway_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
gt_path = "GroundTruth.npy"
pred_path = "AtLoc.npy"

def em_to_poly(path):
	all_res = [["idx", "x", "y", "z", "qw", "qx", "qy", "qz"]]
	points = []
	for i in range(len(os.listdir(path))):
		with open(os.path.join(path, f"{i}.txt")) as f:
			this_num = [float(x) for x in f.read().split()]
		points.append(this_num[ : 3])
		all_res.append([i] + this_num)
	points = np.array(points)
	poly = pv.lines_from_points(points[ : 108])
	tube = poly.tube(radius = 0.5)
	"""
	with open("emcoords.csv", "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(all_res)
	"""
	return tube

def main():
	airway = pv.read(airway_path)
	p = pv.Plotter()
	p.add_mesh(airway, opacity = 0.5)
	gt_traj, pred_traj = np.load(gt_path), np.load(pred_path)
	print(gt_traj.shape, pred_traj.shape)
	for i in range(len(gt_traj)):
		print(gt_traj[i, : 3])
		ln = pv.Line(gt_traj[i, : 3], pred_traj[i, : 3])
		p.add_mesh(ln, color = "red", line_width = 3)
	p.show()

if __name__ == '__main__':
	main()
