import os
import csv
import numpy as np
import pyvista as pv

airway_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
em_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\EM"

def em_to_poly(path):
	poly = pv.lines_from_points(points)
	tube = poly.tube(radius = 0.5)
	return tube

def main():
	airway = pv.read(airway_path)
	p = pv.Plotter(notebook = False, off_screen = True)
	p.add_mesh(airway, opacity = 0.5)
	p.open_gif("emtraj.gif")
	path = em_path
	points = []
	for i in range(len(os.listdir(path))):
		with open(os.path.join(path, f"{i}.txt")) as f:
			this_num = [float(x) for x in f.read().split()]
		points.append(this_num[ : 3])
	points = np.array(points)

	poly = pv.lines_from_points(points[ : 5])
	tube = poly.tube(radius = 0.5)
	p.add_mesh(tube, color = "red")
	for i in range(5, len(points)):
		poly = pv.lines_from_points(points[ : i])
		tube = poly.tube(radius = 0.5)
		p.update_coordinates(tube.points, render = False)
		p.write_frame()
	p.close()

	"""
	tube = em_to_poly(em_path)
	p.add_mesh(tube, color = "red")
	p.show()
	"""

if __name__ == '__main__':
	main()
