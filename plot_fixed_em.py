import os
import csv
import sys
import numpy as np
import pyvista as pv
from scipy.signal import savgol_filter

from utils.misc import str_to_arr

airway_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"

def em_to_poly(path):
	all_res = [["idx", "x", "y", "z", "qw", "qx", "qy", "qz"]]
	points = []
	for i in range(len(os.listdir(path))):
		with open(os.path.join(path, f"{i}.txt")) as f:
			this_num = [float(x) for x in f.read().split()]
		points.append(this_num[ : 3])
		all_res.append([i] + this_num)
	points = np.array(points)
	poly = pv.lines_from_points(points)
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
	em_idx = int(sys.argv[1])
	points = []
	orients = []
	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if int(row["em_idx"]) == em_idx:
				points.append(str_to_arr(row["position"]))
				orients.append(str_to_arr(row["orientation"]))
	points = np.stack(points, axis = 0)
	orients = np.stack(orients, axis = 0)
	points = savgol_filter(points, window_length = 7, polyorder = 2, axis = 0)
	orients = savgol_filter(orients, window_length = 7, polyorder = 2, axis = 0)
	poly = pv.lines_from_points(filtered_points)
	tube = poly.tube(radius = 0.5)
	p.add_mesh(tube, color = "red")
	p.show()

if __name__ == '__main__':
	main()
