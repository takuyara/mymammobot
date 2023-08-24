import os
import csv
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.pose_utils import compute_rotation_quaternion
from ds_gen.camera_features import camera_params
from pose_fixing.move_camera import reverse_to_geno
from utils.cl_utils import load_all_cls, locate_on_cl, get_cl_radius, get_cl_direction
from utils.misc import str_to_arr

from vtkmodules.vtkRenderingCore import vtkCamera


def main():
	all_cls = load_all_cls("./CL")

	img2lb = {}
	with open("select_result_contour_non_interp.csv", newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			em_idx, try_idx, img_idx, label = int(row[0]), int(row[1]), int(row[2]), int(row[3])
			img2lb[(em_idx, img_idx, try_idx)] = label

	rows = [["em_idx", "img_idx", "try_idx", "cont_sim", "corr_sim", "human_eval", "position", "orientation", "up", "cl_idx", "on_line_idx", "lumen_radius", "orient_rot", "orient_norm", "radial_rot", "radial_norm", "axial_norm", "up_rot"]]
	with open("reg_params_non_interp.csv", newline = "") as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
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

			cl_idx, on_line_idx = locate_on_cl(position, all_cls)
			lumen_radius = get_cl_radius(position, all_cls, (cl_idx, on_line_idx))
			cl_orientation = get_cl_direction(all_cls, (cl_idx, on_line_idx))

			cl_offsets = reverse_to_geno(all_cls[cl_idx][0][on_line_idx, ...], cl_orientation, position, orientation, up)

			label = img2lb.get((em_idx, img_idx, try_idx), -1)
			rows.append([em_idx, img_idx, try_idx, cont_sim, corr_sim, label, position, orientation, up, cl_idx, on_line_idx, lumen_radius, *cl_offsets])

	with open("reg_params_non_interp_processed.csv", "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(rows)

if __name__ == '__main__':
	main()
