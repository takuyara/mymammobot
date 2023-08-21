import os
import cv2
import sys
import numpy as np

from pose_fixing.similarity import get_dark_threshold, draw_contours

em_base_path = "./depth-images"

def find_optim_thres(em_idx, bins, rate):
	em_path = os.path.join(em_base_path, f"EM-rawdep-{em_idx}")
	out_path = os.path.join(em_base_path, f"EM-thres-sel-{em_idx}-{bins}-{rate}")
	os.makedirs(out_path, exist_ok = True)
	for i in range(0, len(os.listdir(em_path)), 8):
		rd = np.load(os.path.join(em_path, f"{i:06d}.npy"))
		thres = get_dark_threshold(rd, bins = bins, rate = rate)
		quantile = np.sum(rd < thres) / len(rd.ravel())
		out_img = draw_contours(rd, quantile)
		cv2.imwrite(os.path.join(out_path, f"{i:06d}.png"), out_img)

if __name__ == '__main__':
	find_optim_thres(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
