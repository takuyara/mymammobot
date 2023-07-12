import os
import csv
import numpy as np
import pyvista as pv

airway_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
em_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\EM"
#em_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_17-08-08_Phantom_2\\EM"
#em_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_17-12-48_Phantom_3\\EM"

def em_to_poly(path):
	all_res = [["idx", "x", "y", "z", "qw", "qx", "qy", "qz"]]
	points = []
	for i in range(len(os.listdir(path))):
		with open(os.path.join(path, f"{i}.txt")) as f:
			this_num = [float(x) for x in f.read().split()]
		points.append(this_num[ : 3])
		all_res.append([i] + this_num)
	points = np.array(points)
	poly = pv.lines_from_points(points[150 : 200])
	tube = poly.tube(radius = 0.5)
	with open("emcoords.csv", "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(all_res)
	return tube

def main():
	airway = pv.read(airway_path)
	p = pv.Plotter()
	p.add_mesh(airway, opacity = 0.5)
	tube = em_to_poly(em_path)
	p.add_mesh(tube, color = "red")
	p.show()

if __name__ == '__main__':
	main()
