import os
import csv
import shutil
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from utils.pose_utils import compute_rotation_quaternion
from ds_gen.camera_features import camera_params
from pose_fixing.move_camera import reverse_to_geno, get_radial_axial_offsets
from utils.cl_utils import load_all_cls, find_all_possible_cls, get_direction_dist_radius, index2point, load_all_cls_npy, locate_on_cl
from utils.misc import str_to_arr

from vtkmodules.vtkRenderingCore import vtkCamera


def main():
	all_cls = load_all_cls_npy("./seg_cl_1")

	sum_dists = []
	for i, (points, __) in enumerate(all_cls):
		this_dists = []
		td = 0
		for j in range(len(points)):
			if j > 0:
				td = td + np.linalg.norm(points[j, ...] - points[j - 1, ...])
			this_dists.append(td)
		sum_dists.append(this_dists)

	base_path = "./real_dataset"
	out_path = "./real_dataset_by"
	for this_type in ["confirmed", "all"]:
		for i in range(3):
			this_path = os.path.join(base_path, this_type, f"real-{i}")
			new_path = os.path.join(out_path, this_type, f"real-{i}")
			os.makedirs(new_path, exist_ok = True)
			for this_file in os.listdir(this_path):
				if this_file.endswith(".txt"):
					pose = np.loadtxt(os.path.join(this_path, this_file))
					position = pose[0]
					cl_indices = locate_on_cl(position, all_cls)
					cl_base_point = index2point(all_cls, cl_indices)
					cl_orientation, __, lumen_radius = get_direction_dist_radius(all_cls, cl_indices)
					radial_rot, radial_norm, axial_norm = get_radial_axial_offsets(cl_base_point, cl_orientation, position)
					out_pose = [[cl_indices[0], sum_dists[cl_indices[0]][cl_indices[1]] + axial_norm, radial_norm, radial_rot]]
					np.savetxt(os.path.join(new_path, this_file.replace(".txt", "_clbase.txt")), out_pose, fmt = "%.6f")

	exit()

	with open("reg_params_interp_by.csv", newline = "") as f:
		reader = csv.reader(f)
		for i, row in enumerate(tqdm(reader)):
			if len(row) < 3:
				continue
			em_idx, img_idx, try_idx = int(row[0]), int(row[1]), int(row[2])
			cont_sim, corr_sim = float(row[3]), float(row[4])
			position = str_to_arr(row[5])
			orientation = str_to_arr(row[6])
			up = str_to_arr(row[7])
			focal_point = position + camera_params["focal_length"] * orientation
			camera = vtkCamera()
			camera.SetPosition(*position)
			camera.SetFocalPoint(*focal_point)
			camera.SetViewUp(*up)
			camera.OrthogonalizeViewUp()
			up = camera.GetViewUp()
			orientation = orientation / np.linalg.norm(orientation)
			up = up / np.linalg.norm(up)

			min_f_radial_norm = 1e10

			for cl_indices in find_all_possible_cls(position, all_cls):
				cl_base_point = index2point(all_cls, cl_indices)
				cl_orientation, __, lumen_radius = get_direction_dist_radius(all_cls, cl_indices)

				fixed_focal_length = camera_params["focal_length"] / np.dot(cl_orientation, orientation)
				fixed_focal_point = position + orientation * fixed_focal_length
				f_radial_rot, f_radial_norm, f_axial_norm = get_radial_axial_offsets(cl_base_point, cl_orientation, fixed_focal_point)

				if f_radial_norm < min_f_radial_norm:
					min_f_radial_norm = f_radial_norm
					best_cl_indices = cl_indices

			cl_base_point = index2point(all_cls, best_cl_indices)
			cl_orientation, __, lumen_radius = get_direction_dist_radius(all_cls, best_cl_indices)
			cl_offsets = reverse_to_geno(cl_base_point, cl_orientation, position, orientation, up)	

			label = img2lb.get((em_idx, img_idx, try_idx), -1)
			rows.append([em_idx, img_idx, try_idx, cont_sim, corr_sim, label, position, orientation, up, *cl_indices, lumen_radius, *cl_offsets, min_f_radial_norm])

	with open("reg_params_interp_by_processed.csv", "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(rows)

if __name__ == '__main__':
	main()
