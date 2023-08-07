import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.stats import get_num_bins
from domain_transfer.similarity import kl_sim, corr_sim, mi_sim
from domain_transfer.alignment import reg_depth_maps

em_base_path = "./depth-images/"
show_n_samples = 100
out_img_dir = "./corr_vis"
centrelines = [0]
out_csv_name = "sim_list_xneg.csv"

def write_sim_list():
	all_res = [["em_idx", "img_idx", "kl_sim", "corr_sim", "mi_sim"]]
	for em_idx in centrelines:
		real_folder = os.path.join(em_base_path, f"EM-rawdep-{em_idx}")
		virtual_folder = os.path.join(em_base_path, f"EM-virtual-{em_idx}")
		for i in tqdm(range(len(os.listdir(real_folder)))):
			rd = np.load(os.path.join(real_folder, f"{i:06d}.npy"))
			vd = np.load(os.path.join(virtual_folder, f"{i:06d}.npy"))
			all_res.append([em_idx, i, kl_sim(rd, vd), corr_sim(rd, vd), mi_sim(rd, vd)])
	with open(out_csv_name, "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(all_res)

def plot_sim_list():
	with open(out_csv_name, newline = "") as f:
		reader = csv.DictReader(f)
		kl_sims, corr_sims, mi_sims = [], [], []
		for row in reader:
			kl_sims.append(float(row["kl_sim"]))
			corr_sims.append(float(row["corr_sim"]))
			mi_sims.append(float(row["mi_sim"]))
	"""
	plt.subplot(1, 3, 1)
	plt.scatter(kl_sims, corr_sims)
	plt.title("KL vs Corr")
	plt.subplot(1, 3, 2)
	plt.scatter(corr_sims, mi_sims)
	plt.title("Corr vs MI")
	plt.subplot(1, 3, 3)
	plt.scatter(mi_sims, kl_sims)
	plt.title("MI vs KL")
	plt.show()
	"""
	plt.subplot(1, 3, 1)
	plt.hist(kl_sims, 50)
	plt.title("KL")
	plt.subplot(1, 3, 2)
	plt.hist(corr_sims, 50)
	plt.title("Corr")
	plt.subplot(1, 3, 3)
	plt.hist(mi_sims, 50)
	plt.title("MI")
	plt.show()

def plot_corr_mega():
	with open(out_csv_name, newline = "") as f:
		reader = csv.DictReader(f)
		img_data = []
		for row in reader:
			if int(row["em_idx"]) in centrelines:
				img_data.append((float(row["corr_sim"]), int(row["em_idx"]), int(row["img_idx"])))
	img_data.sort(reverse = True)
	os.makedirs(out_img_dir, exist_ok = True)
	for prefix, prt_arr in [("best_match", img_data[ : show_n_samples]), ("worst_match", img_data[ : -(show_n_samples + 1) : -1])]:
		for out_idx, (corr_sim_v, em_idx, img_idx) in enumerate(prt_arr):
			rd = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
			vd = np.load(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.npy"))
			r_rgb = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"))
			v_rgb = cv2.imread(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.png"))
			#vd_ = reg_depth_maps(vd, rd)
			plt.subplot(2, 4, 1)
			plt.imshow(rd, cmap = "gray")
			plt.colorbar()
			plt.title("Real Depth Map")
			plt.subplot(2, 4, 2)
			plt.imshow(vd, cmap = "gray")
			plt.colorbar()
			plt.title("Virtual Depth Map")
			plt.subplot(2, 4, 3)
			plt.imshow(r_rgb)
			plt.title("Real RGB frame")
			plt.subplot(2, 4, 4)
			plt.imshow(v_rgb)
			plt.title("Virtual RGB frame")

			"""
			plt.imshow(vd_, cmap = "gray")
			plt.colorbar()
			plt.title("Registered Virtual Depth Map")
			"""
			plt.subplot(2, 4, 5)
			plt.hist(rd.flatten(), get_num_bins(rd.flatten()))
			plt.title("Real Depth Histogram")
			plt.subplot(2, 4, 6)
			plt.hist(vd.flatten(), get_num_bins(vd.flatten()))
			plt.title("Virtual Depth Histogram")
			
			plt.subplot(2, 4, 7)
			plt.scatter(rd.flatten(), vd.flatten())
			plt.title("Scatter plot for correlation inspection")

			title = f"{prefix}-{out_idx:03d}-corr-{corr_sim_v:.4f}-EM-{em_idx}-{img_idx}"
			#title = f"{prefix}-{out_idx:03d}-corr-{corr_sim_v:.4f}-corr-after{corr_sim(rd, vd_):.4f}-EM-{em_idx}-{img_idx}"
			plt.suptitle(title)
			#plt.savefig(os.path.join(out_img_dir, title + ".png"))
			plt.show()
			plt.clf()

if __name__ == '__main__':
	#write_sim_list()
	#plot_sim_list()
	plot_corr_mega()
