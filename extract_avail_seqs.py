import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from utils.geometry import arbitrary_perpendicular_vector, rotate_single_vector, get_vector_angle
from utils.cl_utils import load_all_cls, index2point
from utils.misc import str_to_arr

step_size = 2

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

	all_seqs = []
	for em_idx, n_imgs in enumerate([2247, 2575, 2156]):
		waitlist = []
		prev_idx = None
		for img_idx in range(0, n_imgs, step_size):
			if not (em_idx, img_idx) in id2data:
				if len(waitlist) > 0:
					all_seqs.append(waitlist)
					print(len(waitlist))
					waitlist = []
			else:
				waitlist.append((em_idx, img_idx))
		if len(waitlist) > 0:
			all_seqs.append(waitlist)
			print(len(waitlist))

if __name__ == '__main__':
	main()
