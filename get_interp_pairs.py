import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from utils.geometry import arbitrary_perpendicular_vector, rotate_single_vector, get_vector_angle
from utils.cl_utils import load_all_cls, index2point
from utils.misc import str_to_arr

step_size = 1

def get_dist_weighted_mid(d1, d2):
	def dist_weighted_mid(v1, v2, dtype = "vector"):
		if dtype != "angle":
			vres = (v1 * d2 + v2 * d1) / (d1 + d2)
			if dtype != "orientation":
				vdst = np.sum((v1 - v2) ** 2) ** 0.5 / 2
			else:
				vdst = np.sin(np.deg2rad(get_vector_angle(v1, v2)))
		else:
			v1, v2 = np.deg2rad(v1), np.deg2rad(v2)
			sn = (np.sin(v1) * d2 + np.sin(v2) * d1) / (d1 + d2)
			cs = (np.cos(v1) * d2 + np.cos(v2) * d1) / (d1 + d2)
			vres = np.rad2deg(np.arctan2(sn, cs))
			vdst = np.abs(v1 - v2)
			while vdst > 360:
				vdst -= 360
		return vres, vdst
	return dist_weighted_mid

def main():
	id2data = {}
	with open("reg_params_non_interp_processed.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if int(row["human_eval"]) == 1:
				id_tup = int(row["em_idx"]), int(row["img_idx"])
				if id_tup in id2data:
					print(f"Warning: duplicated correct data for {id_tup}.")
				id2data[id_tup] = row

	interp_pairs = []
	all_lens = []
	for em_idx, n_imgs in enumerate([2247, 2575, 2156]):
		waitlist = []
		prev_idx = None
		for img_idx in range(0, n_imgs, step_size):
			if not (em_idx, img_idx) in id2data:
				waitlist.append(img_idx)
			else:
				for t_img_idx in waitlist:
					interp_pairs.append((em_idx, t_img_idx, prev_idx, img_idx))
					if img_idx - prev_idx > 100:
						print(em_idx, prev_idx, img_idx)
					all_lens.append(img_idx - prev_idx - 1)
				prev_idx = img_idx
				waitlist = []
		print("Discarded # frames: ", len(waitlist))


	all_cls = load_all_cls("./CL")
	res = [["em_idx", "img_idx", "position", "orientation", "up_rot", "orientation_scale", "radial_scale", "axial_scale", "up_rot_scale"]]
	for em_idx, img_idx, prev_idx, succ_idx in interp_pairs:
		prev_dt = id2data[(em_idx, prev_idx)]
		succ_dt = id2data[(em_idx, succ_idx)]
		dist_weighted_mid = get_dist_weighted_mid(img_idx - prev_idx, succ_idx - img_idx)
		cl_mid_point, cl_scale = dist_weighted_mid(index2point(all_cls, (int(prev_dt["cl_idx"]), int(prev_dt["on_line_idx"]))), index2point(all_cls, (int(succ_dt["cl_idx"]), int(succ_dt["on_line_idx"]))))
		orient_mid, orient_scale = dist_weighted_mid(str_to_arr(prev_dt["orientation"]), str_to_arr(succ_dt["orientation"]), dtype = "orientation")
		orient_mid = orient_mid / np.linalg.norm(orient_mid)
		radial_norm_mid, radial_norm_scale = dist_weighted_mid(float(prev_dt["radial_norm"]), float(succ_dt["radial_norm"]), dtype = "norm")
		radial_rot_mid, radial_rot_scale = dist_weighted_mid(float(prev_dt["radial_rot"]), float(succ_dt["radial_rot"]), dtype = "angle")
		up_rot_mid, up_rot_scale = dist_weighted_mid(float(prev_dt["up_rot"]), float(succ_dt["up_rot"]), dtype = "angle")
		final_mid_point = cl_mid_point + rotate_single_vector(arbitrary_perpendicular_vector(orient_mid), orient_mid, radial_rot_mid) * radial_norm_mid
		res.append([em_idx, img_idx, final_mid_point, orient_mid, up_rot_mid, orient_scale, radial_norm_scale, cl_scale, up_rot_scale])

	with open("interp_data.csv", "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(res)

if __name__ == '__main__':
	main()
