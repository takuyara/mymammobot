import csv
import numpy as np
from utils.misc import str_to_arr

with open("aggred_res.csv", newline = "") as f:
	reader = csv.DictReader(f)
	for row in reader:
		em_idx, img_idx = int(row["em_idx"]), int(row["img_idx"])
		position = str_to_arr(row["position"])
		orientation = str_to_arr(row["orientation"])
		up = str_to_arr(row["up"])
		out_path = f"./real_dataset/all/real-{em_idx}/{img_idx:06d}.txt"
		np.savetxt(out_path, np.stack([position, orientation, up], axis = 0), fmt = "%.6f")