import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.stats import get_num_bins
from domain_transfer.similarity import kl_sim, corr_sim, mi_sim, comb_corr_sim, dark_threshold_v, dark_threshold_r
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

def plot_bf_fix(em_idx, try_idx = 0):
	fix_path = os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}")
	for fixed_map_path in os.listdir(fix_path):
		if not fixed_map_path.endswith(".npy"):
			continue
		img_idx = int(fixed_map_path.replace(".npy", ""))
		rd = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
		vd = np.load(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.npy"))
		fd = np.load(os.path.join(fix_path, fixed_map_path))
		r_rgb = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"))
		v_rgb = cv2.imread(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.png"))
		f_rgb = cv2.imread(os.path.join(fix_path, fixed_map_path.replace(".npy", ".png")))
		#vd_ = reg_depth_maps(vd, rd)
		corr_vd, corr_fd = comb_corr_sim(rd, vd), comb_corr_sim(rd, fd)
		plt.subplot(3, 4, 1)
		plt.imshow(rd, cmap = "gray")
		plt.colorbar()
		plt.title("Real Depth Map")
		plt.subplot(3, 4, 2)
		plt.imshow(vd, cmap = "gray")
		plt.colorbar()
		plt.title(f"Virtual Depth Map {corr_vd:.4f}")
		plt.subplot(3, 4, 3)
		plt.imshow(fd, cmap = "gray")
		plt.colorbar()
		plt.title(f"Fixed Virtual Depth Map {corr_fd:.4f}")
		plt.subplot(3, 4, 4)
		plt.imshow(r_rgb)
		plt.title("Real RGB Frame")
		plt.subplot(3, 4, 5)
		plt.imshow(v_rgb)
		plt.title("Virtual RGB Frame")
		plt.subplot(3, 4, 6)
		plt.imshow(f_rgb)
		plt.title("Fixed RGB Frame")
		plt.subplot(3, 4, 7)
		plt.scatter(rd.flatten(), vd.flatten())
		plt.xlabel("SFS depth")
		plt.ylabel("Virtual depth")
		plt.title("Scatter: not fixed")
		plt.subplot(3, 4, 8)
		plt.scatter(rd.flatten(), fd.flatten())
		plt.xlabel("SFS depth")
		plt.ylabel("Virtual depth")
		plt.title("Scatter: fixed")
		plt.subplot(3, 4, 9)
		plt.imshow((rd > dark_threshold_r).astype(np.uint8))
		plt.title("SFS Mask")
		plt.subplot(3, 4, 10)
		plt.imshow((vd > dark_threshold_v).astype(np.uint8))
		#oc_d = adjusted_corr(rd[vd <= dark_threshold].flatten(), vd[vd <= dark_threshold].flatten())
		#oc_l = adjusted_corr(rd[vd > dark_threshold].flatten(), vd[vd > dark_threshold].flatten())
		#plt.title(f"Original Mask D {oc_d:.4f} vs L {oc_l:.4f}")
		plt.title("Original Mask")
		plt.subplot(3, 4, 11)
		plt.imshow((fd > dark_threshold_v).astype(np.uint8))
		#fc_d = adjusted_corr(rd[fd <= dark_threshold].flatten(), fd[fd <= dark_threshold].flatten())
		#fc_l = adjusted_corr(rd[fd > dark_threshold].flatten(), fd[fd > dark_threshold].flatten())
		#plt.title(f"Fixed Mask D {fc_d:.4f} vs L {fc_l:.4f}")
		plt.title("Fixed Mask")

		plt.suptitle(f"EM-{em_idx}-{img_idx}")
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
			plt.title("Real RGB Frame")
			plt.subplot(2, 4, 4)
			plt.imshow(v_rgb)
			plt.title("Virtual RGB fFrame")

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
			plt.title("Scatter Plot for Correlation")

			title = f"{prefix}-{out_idx:03d}-corr-{corr_sim_v:.4f}-EM-{em_idx}-{img_idx}"
			#title = f"{prefix}-{out_idx:03d}-corr-{corr_sim_v:.4f}-corr-after{corr_sim(rd, vd_):.4f}-EM-{em_idx}-{img_idx}"
			plt.suptitle(title)
			#plt.savefig(os.path.join(out_img_dir, title + ".png"))
			plt.show()
			plt.clf()

if __name__ == '__main__':
	#write_sim_list()
	#plot_sim_list()
	#plot_corr_mega()
	plot_bf_fix(0, 1)
