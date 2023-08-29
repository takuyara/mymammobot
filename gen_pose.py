import csv
import numpy as np
import shutil
from utils.misc import str_to_arr

with open("aggred_res.csv", newline = "") as f:
	reader = csv.DictReader(f)
	for row in reader:
		em_idx, img_idx, human_eval = int(row["em_idx"]), int(row["img_idx"]), int(row["human_eval"])
		if human_eval == 1:
			position = str_to_arr(row["position"])
			orientation = str_to_arr(row["orientation"])
			up = str_to_arr(row["up"])
			out_path = f"./real_dataset/confirmed/real-{em_idx}/{img_idx:06d}.txt"
			np.savetxt(out_path, np.stack([position, orientation, up], axis = 0), fmt = "%.6f")
			npy_path = f"./depth-images/EM-rawdep-{em_idx}/{img_idx:06d}.npy"
			shutil.copy(npy_path, out_path.replace(".txt", ".npy"))