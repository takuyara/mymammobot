import csv
import numpy as np
import pyvista as pv

airway_path = "./meshes/Airway_Phantom_AdjustSmooth.stl"
cl_paths = ["./CL/CL0.dat", "./CL/CL1.dat", "./CL/CL2.dat", "./CL/CL3.dat", "./CL/CL4.dat"]
#n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n  
def cl_to_poly(path):
	points = []
	scalars = []
	with open(path, newline = "") as f:
		reader = csv.DictReader(f, delimiter = " ")
		for row in reader:
			points.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
			scalars.append(float(row["MaximumInscribedSphereRadius"]))
	points = np.array(points)
	scalars = np.array(scalars)
	poly = pv.lines_from_points(points)
	poly["scalars"] = scalars
	tube = poly.tube(radius = 0.5)
	return tube

def main():
	airway = pv.read(airway_path)
	p = pv.Plotter()
	p.add_mesh(airway, opacity = 0.5)
	for path in cl_paths:
		tube = cl_to_poly(path)
		p.add_mesh(tube, color = "red")
	p.show()

if __name__ == '__main__':
	main()
