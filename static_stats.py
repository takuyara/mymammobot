import csv
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from utils.geometry import get_vector_angle
from utils.cl_utils import load_all_cls, get_direction_dist_radius
from utils.misc import str_to_arr

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data-path", type = str, default = "./aggred_res.csv")
	parser.add_argument("--uses-interp", action = "store_true", default = False)
	parser.add_argument("--num-bins", type = int, default = 6)
	parser.add_argument("--out-path", type = str, default = "./ds_gen/static_distrib.json")
	return parser.parse_args()

def main():
	args = get_args()
	all_cls = load_all_cls("./CL")
	radiuses, r_norms, fr_norms, angles = [], [], [], []
	rates = []
	with open(args.data_path, newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if args.uses_interp or row["human_eval"] == "1":
				radiuses.append(float(row["lumen_radius"]))
				r_norms.append(float(row["radial_norm"]))
				fr_norms.append(float(row["focal_radial_norm"]))
				cl_direction = get_direction_dist_radius(all_cls, (int(row["cl_idx"]), int(row["on_line_idx"])))[0]
				orientation = str_to_arr(row["orientation"])
				angle = get_vector_angle(orientation, cl_direction)
				angles.append(angle)
	r_mn, r_mx = np.min(radiuses), np.max(radiuses)
	bin_width = (r_mx - r_mn) / args.num_bins
	#plt.hist(radiuses, bins = args.num_bins)
	#plt.show()
	rnorm_distrib, frnorm_distrib = [[] for __ in range(args.num_bins)], [[] for __ in range(args.num_bins)]
	for t_radius, t_rnorm, t_frnorm in zip(radiuses, r_norms, fr_norms):
		idx = int(min((t_radius - r_mn) // bin_width, args.num_bins - 1))
		rnorm_distrib[idx].append(t_rnorm)
		frnorm_distrib[idx].append(t_frnorm)
	res = {"num_bins ": args.num_bins, "bin_width": bin_width, "min_radius": r_mn, "rnorm_distrib": rnorm_distrib, "frnorm_distrib": frnorm_distrib}
	json.dump(res, open(args.out_path, "w"))

if __name__ == '__main__':
	main()
