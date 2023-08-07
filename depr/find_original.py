import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import csv
import os

focal_length = 10

em_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\EM"
cl_paths = ["./CL/CL0.dat", "./CL/CL4.dat"]

all_points = []
for cl_path in cl_paths:
	points = []
	with open(cl_path, newline = "") as f:
		reader = csv.DictReader(f, delimiter = " ")
		for row in reader:
			points.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
	points.reverse()
	all_points.append(np.array(points))

orig_vecs = []
for this_em_path in tqdm(os.listdir(em_path)):
	general_pose = np.loadtxt(os.path.join(em_path, this_em_path)).reshape(-1)
	position = general_pose[ : 3]
	rotation = R.from_quat(general_pose[3 : ])
	min_dist = 1e100
	for points in all_points:
		dists = np.sum((points[ : -1, : ] - position) ** 2, axis = 1)
		cl_idx, cl_dist = np.argmin(dists), np.min(dists)
		if cl_dist < min_dist:
			min_dist = cl_dist
			min_ori = points[cl_idx + 1] - points[cl_idx - 1]
			min_pt = points[cl_idx]
	min_ori = min_ori / np.linalg.norm(min_ori)
	orig_vec = rotation.apply(min_ori, inverse = True)
	if this_em_path == "10.txt":
		print(min_ori, orig_vec, min_pt)
	orig_vecs.append(orig_vec / np.linalg.norm(orig_vec))
print(np.mean(orig_vecs, axis = 0), np.std(orig_vecs, axis = 0))
