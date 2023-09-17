import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
import shutil
import sys
import pyvista as pv
from utils.misc import str_to_arr
from ds_gen.depth_map_generation import get_depth_map

from utils.stats import get_num_bins
from pose_fixing.similarity import kl_sim, corr_sim, mi_sim, comb_corr_sim, dark_threshold_v, dark_threshold_r, cache_base_data, contour_sim, draw_contours, cache_multiscale_base_data
#from domain_transfer.alignment import reg_depth_maps

em_base_path = "./depth-images/"
show_n_samples = 100
out_img_dir = "./corr_vis"
centrelines = [0]
out_csv_name = "sim_list_xneg.csv"
comp_metadata = "./register_params.csv"
select_result = "./select_result.csv"
try_selects = "./best_trys.csv"

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

def plot_one_fix(em_idx, img_idx, try_idx):
	plt.clf()
	rd = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
	vd = np.load(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.npy"))
	fd = np.load(os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}", f"{img_idx:06d}.npy"))
	r_rgb = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"))
	v_rgb = cv2.imread(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.png"))
	f_rgb = cv2.imread(os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}", f"{img_idx:06d}.png"))
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
	plt.title("Original Mask")
	plt.subplot(3, 4, 11)
	plt.imshow((fd > dark_threshold_v).astype(np.uint8))
	plt.title("Fixed Mask")
	plt.suptitle(f"EM-{em_idx}-{img_idx}")

def plot_bf_fix(em_idx, try_idx = 0):
	fix_path = os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}")
	eval_path = os.path.join(em_base_path, f"EM-virtual-autofix-eval-{em_idx}-{try_idx}")
	os.makedirs(eval_path, exist_ok = True)
	for fixed_map_path in os.listdir(fix_path):
		if not fixed_map_path.endswith(".npy"):
			continue
		img_idx = int(fixed_map_path.replace(".npy", ""))
		plot_one_fix(em_idx, img_idx, try_idx)
		plt.savefig(os.path.join(eval_path, f"{img_idx:06d}.png"))
		plt.clf()


def plot_max_depth():
	all_rds, all_fds = [], []
	with open(try_selects, newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			em_idx = int(row[0])
			img_idx = int(row[1])
			try_idx = int(row[2])
			if try_idx != -1:
				rd = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
				fd = np.load(os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}", f"{img_idx:06d}.npy"))
				all_rds.append(rd.max())
				all_fds.append(fd.max())
				if fd.max() > 100:
					print(em_idx, try_idx, img_idx)
	plt.scatter(all_rds, all_fds)
	plt.xlabel("SFS depth")
	plt.ylabel("Virtual depth")
	plt.show()


def plot_fix_maxfit(em_indices):
	best_corrs = {}
	with open(comp_metadata, newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			em_idx, img_idx, this_corr = int(row[0]), int(row[1]), float(row[2])
			if em_idx not in em_indices:
				continue
			if (em_idx, img_idx) not in best_corrs or best_corrs[(em_idx, img_idx)][0] < this_corr:
				best_corrs[(em_idx, img_idx)] = (this_corr, int(row[-1]))
	for (em_idx, img_idx), (__, try_idx) in tqdm(best_corrs.items()):
		plot_one_fix(em_idx, img_idx, try_idx)
		eval_path = os.path.join(em_base_path, f"EM-virtual-autofix-best-{em_idx}")
		os.makedirs(eval_path, exist_ok = True)
		plt.savefig(os.path.join(eval_path, f"{img_idx:06d}.png"))

def find_correspond_tryidx():
	cand_rows = []
	res_rows = []
	with open(comp_metadata, newline = "") as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
			if i == 1775:
				break
			cand_rows.append(row)
	
	nfk = 0
	for row in tqdm(cand_rows):
		rd = np.load(os.path.join(em_base_path, f"EM-rawdep-{int(row[0])}", f"{int(row[1]):06d}.npy"))
		resolve_idx = None
		for try_idx in [1, 2]:
			fix_path = os.path.join(em_base_path, f"EM-virtual-autofix-{int(row[0])}-{try_idx}", f"{int(row[1]):06d}.npy")
			sim = comb_corr_sim(rd, np.load(fix_path))
			#print(sim, float(row[2]))
			if abs(sim - float(row[2])) < 1e-5:
				resolve_idx = try_idx
		if resolve_idx is None:
			nfk += 1
			continue
		res_rows.append(row + [resolve_idx])

	print("Mismatch count: ", nfk)

	with open(comp_metadata.replace(".csv", "_cgd.csv"), "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(res_rows)

	# copy and fix: 5388+

def select_between_candidates(em_idx, try_indices, out_index):
	key2adj = {"q": "trans", "w": "orient", "e": "up", "r": "wrong", "a": "available"}
	fix_path = os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_indices[0]}")
	out_path = os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{out_index}")
	os.makedirs(out_path, exist_ok = True)
	img_indices = []
	for fixed_map_path in os.listdir(fix_path):
		if fixed_map_path.endswith(".npy"):
			img_indices.append(int(fixed_map_path.replace(".npy", "")))
	fig, ax = plt.subplots()
	for img_idx in img_indices:
		# 0~528 no right/wrong info, have correction info.
		# 536~916 no right/wrong info, no correction info.
		# 916+: have wrong info, no correct / need modify info, no correction info.
		plt.clf()
		plt.subplot(3, 4, 1)
		r_rgb = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"))
		plt.imshow(r_rgb)
		plt.title("Real Frame")
		for i, try_idx in enumerate(try_indices):
			plt.subplot(3, 4, 5 + i)
			f_rgb = cv2.imread(os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}", f"{img_idx:06d}.png"))
			plt.imshow(f_rgb)
			plt.title(f"Choice {i + 1}")
		plt.suptitle(f"EM {em_idx} {img_idx}")
		fig.canvas.draw()
		img_arr = np.array(fig.canvas.renderer.buffer_rgba())
		img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
		img_arr = cv2.resize(img_arr, (1200, 900))
		best_choice, best_adj = None, []
		while True:
			cv2.imshow("img_select", img_arr)
			key = cv2.waitKey(0) & 0xFF
			for i in range(len(try_indices)):
				if key == ord(f"{i + 1}"):
					best_choice = i
					#print("Best choice: ", i)
			if key == ord("`"):
				best_choice = -1
				break
			if best_choice is not None:
				if key == ord("r"):
					best_adj.append("need_modify")
					break
				elif key == ord("t"):
					best_adj.append("can_be_used")
					break
			"""
			if best_choice is not None:
				print("In search adj")
				for adj in ["q", "w", "e", "r", "a"]:
					if key == ord(adj):
						best_adj.append(key2adj[adj])
						print("Added adj")
				if len(best_adj) > 0 and key == ord("t"):
					print("Quit.")
					break
			"""
		print(best_choice, best_adj)
		src_path = os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_indices[best_choice]}")
		shutil.copy(os.path.join(src_path, f"{img_idx:06d}.png"), os.path.join(out_path, f"{img_idx:06d}.png"))
		shutil.copy(os.path.join(src_path, f"{img_idx:06d}.npy"), os.path.join(out_path, f"{img_idx:06d}.npy"))
		with open(try_selects, "a", newline = "") as f:
			writer = csv.writer(f)
			writer.writerow([em_idx, img_idx, try_indices[best_choice] if best_choice != -1 else -1, try_indices] + best_adj)

def manually_select_alignment(em_idx, try_idx):
	# Y, 1: Correctly aligned. Could use small adjustions.
	# N, 0: Nothing alike. Need much larger displacements.
	# R, 2: Looked on the correct direction. Need small adjustment or large rotations.

	fix_path = os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}")
	eval_path = os.path.join(em_base_path, f"EM-virtual-autofix-eval-{em_idx}-{try_idx}")
	y_path = os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}-good")
	n_path = os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}-bad")
	r_path = os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}-rot")
	os.makedirs(y_path, exist_ok = True)
	os.makedirs(n_path, exist_ok = True)
	os.makedirs(r_path, exist_ok = True)

	for out_img_name in os.listdir(eval_path):
		img_idx = int(out_img_name.replace(".png", ""))
		img = cv2.imread(os.path.join(eval_path, out_img_name))
		img = cv2.resize(img, (1200, 900))
		while True:
			cv2.imshow("img_select", img)
			key = cv2.waitKey(0) & 0xFF
			if key == ord("y"):
				out_path = y_path
				op = 1
				break
			elif key == ord("n"):
				out_path = n_path
				op = 0
				break
			elif key == ord("r"):
				out_path = r_path
				op = 2
				break
		shutil.copy(os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}", out_img_name), os.path.join(out_path, out_img_name))
		out_img_name = out_img_name.replace(".png", ".npy")
		shutil.copy(os.path.join(em_base_path, f"EM-virtual-autofix-{em_idx}-{try_idx}", out_img_name), os.path.join(out_path, out_img_name))
		with open(select_result, "a", newline = "") as f:
			writer = csv.writer(f)
			writer.writerow([em_idx, img_idx, try_idx, op])

	cv2.destroyAllWindows()

def manually_select_alignment_contour(em_idx, try_idx):
	# Y, 1: Correctly aligned. Could use small adjustions.
	# N, 0: Nothing alike. Need much larger displacements.

	eval_path = os.path.join(em_base_path, f"EM-newfix-eval-{em_idx}-{try_idx}")
	y_path = os.path.join(em_base_path, f"EM-newfix-eval-{em_idx}-{try_idx}-good")
	n_path = os.path.join(em_base_path, f"EM-newfix-eval-{em_idx}-{try_idx}-bad")
	os.makedirs(y_path, exist_ok = True)
	os.makedirs(n_path, exist_ok = True)

	for out_img_name in os.listdir(eval_path):
		img_idx = int(out_img_name.replace(".png", ""))
		img_path = os.path.join(eval_path, out_img_name)
		processed = False
		for possible_path in [y_path, n_path]:
			if os.path.exists(os.path.join(possible_path, out_img_name)):
				processed = True
				break
		if processed:
			print(f"Skipping EM-{em_idx}-{img_idx} of try {try_idx}.")
			continue
		img = cv2.imread(img_path)
		img = cv2.resize(img, (1200, 900))
		while True:
			cv2.imshow("img_select", img)
			key = cv2.waitKey(0) & 0xFF
			if key == ord("y"):
				out_path = y_path
				op = 1
				break
			elif key == ord("n"):
				out_path = n_path
				op = 0
				break
		
		shutil.copy(os.path.join(eval_path, out_img_name), os.path.join(out_path, out_img_name))
		
		with open("select_result_contour.csv", "a", newline = "") as f:
			writer = csv.writer(f)
			writer.writerow([em_idx, try_idx, img_idx, op])

	cv2.destroyAllWindows()

def manually_select_alignment_interp(em_idx, try_idx):
	# Y, 1: Correctly aligned. Could use small adjustions.
	# N, 0: Nothing alike. Need much larger displacements.

	eval_path = os.path.join(em_base_path, f"EM-interpfix-eval-{em_idx}-{try_idx}")
	y_path = os.path.join(em_base_path, f"EM-interpfix-eval-{em_idx}-{try_idx}-good")
	n_path = os.path.join(em_base_path, f"EM-interpfix-eval-{em_idx}-{try_idx}-bad")
	os.makedirs(y_path, exist_ok = True)
	os.makedirs(n_path, exist_ok = True)

	for out_img_name in os.listdir(eval_path):
		img_idx = int(out_img_name.replace(".png", ""))
		img_path = os.path.join(eval_path, out_img_name)
		processed = False
		for possible_path in [y_path, n_path]:
			if os.path.exists(os.path.join(possible_path, out_img_name)):
				processed = True
				break
		if processed:
			print(f"Skipping EM-{em_idx}-{img_idx} of try {try_idx}.")
			continue
		img = cv2.imread(img_path)
		img = cv2.resize(img, (1200, 900))
		while True:
			cv2.imshow("img_select", img)
			key = cv2.waitKey(0) & 0xFF
			if key == ord("y"):
				out_path = y_path
				op = 1
				break
			elif key == ord("n"):
				out_path = n_path
				op = 0
				break
		
		shutil.copy(os.path.join(eval_path, out_img_name), os.path.join(out_path, out_img_name))
		
		with open("select_result_contour_interp.csv", "a", newline = "") as f:
			writer = csv.writer(f)
			writer.writerow([em_idx, try_idx, img_idx, op])

	cv2.destroyAllWindows()


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

def plot_one_fix_contour(em_idx, img_idx, try_idx):
	plt.clf()
	rd = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
	#vd = np.load(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.npy"))
	fd = np.load(os.path.join(em_base_path, f"EM-newfix-{em_idx}-{try_idx}", f"{img_idx:06d}.npy"))
	r_rgb = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"))
	#v_rgb = cv2.imread(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.png"))
	f_rgb = cv2.imread(os.path.join(em_base_path, f"EM-newfix-{em_idx}-{try_idx}", f"{img_idx:06d}.png"))

	base_contours, quantile, base_light_mask = cache_base_data(rd, rate = 0.1)
	cont_sim = contour_sim(fd, base_contours, quantile)
	r_contour_img = draw_contours(rd, quantile)
	f_contour_img = draw_contours(fd, quantile)
	corr_sim = comb_corr_sim(rd, fd)

	#corr_vd, corr_fd = comb_corr_sim(rd, vd), comb_corr_sim(rd, fd)
	plt.subplot(2, 2, 1)
	plt.imshow(r_rgb)
	plt.title("Real RGB Frame")
	plt.subplot(2, 2, 2)
	plt.imshow(f_rgb)
	plt.title("Fixed RGB Frame")
	plt.subplot(2, 2, 3)
	plt.imshow(r_contour_img)
	plt.title("Real Depth Map")
	plt.subplot(2, 2, 4)
	plt.imshow(f_contour_img)
	plt.title(f"Fixed Depth Map")

	plt.suptitle(f"EM-{em_idx}-{img_idx} {cont_sim:.0f}, {corr_sim:.4f}")

def plot_one_fix_interp(em_idx, img_idx, try_idx):
	plt.clf()
	rd = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
	#vd = np.load(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.npy"))
	fd = np.load(os.path.join(em_base_path, f"EM-interpfix-{em_idx}-{try_idx}", f"{img_idx:06d}.npy"))
	r_rgb = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"))
	#v_rgb = cv2.imread(os.path.join(em_base_path, f"EM-virtual-{em_idx}", f"{img_idx:06d}.png"))
	f_rgb = cv2.imread(os.path.join(em_base_path, f"EM-interpfix-{em_idx}-{try_idx}", f"{img_idx:06d}.png"))

	base_contours, quantile, base_light_mask = cache_base_data(rd, rate = 0.1)
	cont_sim = contour_sim(fd, base_contours, quantile)
	r_contour_img = draw_contours(rd, quantile)
	f_contour_img = draw_contours(fd, quantile)
	corr_sim = comb_corr_sim(rd, fd)

	#corr_vd, corr_fd = comb_corr_sim(rd, vd), comb_corr_sim(rd, fd)
	plt.subplot(2, 2, 1)
	plt.imshow(r_rgb)
	plt.title("Real RGB Frame")
	plt.subplot(2, 2, 2)
	plt.imshow(f_rgb)
	plt.title("Fixed RGB Frame")
	plt.subplot(2, 2, 3)
	plt.imshow(r_contour_img)
	plt.title("Real Depth Map")
	plt.subplot(2, 2, 4)
	plt.imshow(f_contour_img)
	plt.title(f"Fixed Depth Map")

	plt.suptitle(f"EM-{em_idx}-{img_idx} {cont_sim:.0f}, {corr_sim:.4f}")


def draw_contour_outputs(em_idx, try_idx):
	plt.figure(figsize = (20, 15))
	fix_path = os.path.join(em_base_path, f"EM-newfix-{em_idx}-{try_idx}")
	eval_path = os.path.join(em_base_path, f"EM-newfix-eval-{em_idx}-{try_idx}")
	os.makedirs(eval_path, exist_ok = True)
	for fixed_map_path in os.listdir(fix_path):
		if not fixed_map_path.endswith(".npy"):
			continue
		img_idx = int(fixed_map_path.replace(".npy", ""))
		plot_one_fix_contour(em_idx, img_idx, try_idx)
		plt.savefig(os.path.join(eval_path, f"{img_idx:06d}.png"))
		plt.clf()


def draw_interp_outputs(em_idx, try_idx):
	plt.figure(figsize = (20, 15))
	fix_path = os.path.join(em_base_path, f"EM-interpfix-{em_idx}-{try_idx}")
	eval_path = os.path.join(em_base_path, f"EM-interpfix-eval-{em_idx}-{try_idx}")
	os.makedirs(eval_path, exist_ok = True)
	for fixed_map_path in os.listdir(fix_path):
		if not fixed_map_path.endswith(".npy"):
			continue
		img_idx = int(fixed_map_path.replace(".npy", ""))
		plot_one_fix_interp(em_idx, img_idx, try_idx)
		plt.savefig(os.path.join(eval_path, f"{img_idx:06d}.png"))
		plt.clf()


def plot_sim_scatter():
	trys_data = {}
	with open("new_reg_params_from_doc.csv", newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			em_idx, img_idx, try_idx = int(row[0]), int(row[1]), int(row[2])
			cont_sim, corr_sim = float(row[3]), float(row[4])
			if (em_idx, img_idx) in trys_data:
				print(f"Warning: multiple trys ({try_idx}, {trys_data[(em_idx, img_idx)][-1]}) for same image {em_idx}, {img_idx}.")
			trys_data[(em_idx, img_idx)] = (cont_sim, corr_sim, try_idx)
	
	cont_sims, corr_sims = [[], []], [[], []]
	with open("select_result_contour.csv", newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			em_idx, try_idx, img_idx, label = int(row[0]), int(row[1]), int(row[2]), int(row[3])
			cont_sim, corr_sim, try_idx_1 = trys_data[(em_idx, img_idx)]
			if try_idx_1 != try_idx:
				print(f"Warning: wrong trys ({try_idx}, {try_idx_1}) for same image {em_idx}, {img_idx}.")
			cont_sims[label].append(cont_sim)
			corr_sims[label].append(corr_sim)

	"""
	plt.subplot(2, 2, 1)
	plt.hist(cont_sims[0], range = (-35, 0), density = True)
	plt.title("Cont sim Wrong")
	plt.subplot(2, 2, 2)
	plt.hist(cont_sims[1], range = (-35, 0), density = True)
	plt.title("Cont sim Correct")
	plt.subplot(2, 2, 3)
	plt.hist(corr_sims[0], range = (0.7, 1.0), density = True)
	plt.title("Corr sim Wrong")
	plt.subplot(2, 2, 4)
	plt.hist(corr_sims[1], range = (0.7, 1.0), density = True)
	plt.title("Corr sim Correct")
	plt.show()
	"""

	print(f"Cont sim right: {np.mean(cont_sims[1])} wrong: {np.mean(cont_sims[0])}")
	print(f"Corr sim right: {np.mean(corr_sims[1])} wrong: {np.mean(corr_sims[0])}")

def plot_final_fixes(show_fix):
	out_path = "./depth-images/fix-final-confirmed-{}".format("with-gt" if show_fix else "without-gt")
	#out_path = "./depth-images/fix-noproj-GSFonly"
	os.makedirs(out_path, exist_ok = True)
	p = pv.Plotter(off_screen = True, window_size = (224, 224))
	p.add_mesh(pv.read("./meshes/Airway_Phantom_AdjustSmooth.stl"))
	uses_colorbar = False

	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in tqdm(reader):
			plt.clf()
			em_idx, img_idx, human_eval = int(row["em_idx"]), int(row["img_idx"]), int(row["human_eval"])
			if human_eval != 1:
				continue
			position, orientation, up = str_to_arr(row["position"]), str_to_arr(row["orientation"]), str_to_arr(row["up"])
			rgb_gt, dep_gt = get_depth_map(p, position, orientation, up, get_outputs = True)
			rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGBA2GRAY)
			dep_rc = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
			dep_em = np.load(os.path.join(em_base_path, f"EM-virtual-RF-{em_idx}", f"{img_idx:06d}.npy"))
			rgb_rc = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"), cv2.IMREAD_GRAYSCALE)
			rgb_em = cv2.imread(os.path.join(em_base_path, f"EM-virtual-RF-{em_idx}", f"{img_idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
			rgb_rc = rgb_rc[9 : 255, ...]
			#print(rgb_rc.shape)
			rgb_gt = rgb_gt / 255.0
			rgb_em = rgb_em / 255.0
			rgb_rc = rgb_rc / 255.0
			
			if show_fix:
				out_dt = [(rgb_rc, "GSF-RC"), (rgb_em, "GSF-EM"), (rgb_gt, "GSF-GT"), (dep_rc, "DM-RC-SFS"), (dep_em, "DM-EM-Mesh"), (dep_gt, "DM-GT-Mesh")]
				num_plots = 2, 3
			else:
				out_dt = [(rgb_rc, "GSF-RC"), (rgb_em, "GSF-EM"), (dep_rc, "DM-RC-SFS"), (dep_em, "DM-EM-Mesh")]
				num_plots = 2, 2
			
			"""
			out_dt = [(rgb_rc, "GSF-RC"), (rgb_em, "GSF-EM")]
			num_plots = 1, 2
			"""

			for i, (out_img, out_title) in enumerate(out_dt):
				plt.subplot(*num_plots, i + 1)
				if out_title.find("DM") != -1:
					plt.imshow(out_img)
				else:
					plt.imshow(out_img, cmap = "gray", vmin = 0, vmax = 1)
				if uses_colorbar:
					plt.colorbar()
				plt.axis("off")
				plt.title(out_title)
			plt.suptitle(f"REAL-{em_idx}-{img_idx}, {human_eval}")
			#plt.suptitle(f"REAL-{em_idx}, Frame-{img_idx}")
			#plt.show()
			plt.savefig(os.path.join(out_path, f"{em_idx}-{img_idx:04d}.png"))

def plot_out_img(proj, show_fix):
	chosen_data = [(0, 24), (0, 644), (0, 822), (0, 1058), (1, 1926), (1, 2404), (2, 1320)]
	out_path = "./depth-images/fix-final-selected-{}".format("with-gt" if show_fix else "without-gt")
	os.makedirs(out_path, exist_ok = True)
	p = pv.Plotter(off_screen = True, window_size = (224, 224))
	p.add_mesh(pv.read("./meshes/Airway_Phantom_AdjustSmooth.stl"))
	uses_colorbar = False
	proj_ident = "RF" if not proj else "PROJ"

	plt.figure(figsize = (6, 3.5))
	plt.subplots_adjust(left = 0.01, right = 0.99, top = 0.99, bottom = 0.04, hspace = 0.05, wspace = 0.05)
	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in tqdm(reader):
			plt.clf()
			em_idx, img_idx, human_eval = int(row["em_idx"]), int(row["img_idx"]), int(row["human_eval"])
			if (em_idx, img_idx) not in chosen_data:
				continue
			position, orientation, up = str_to_arr(row["position"]), str_to_arr(row["orientation"]), str_to_arr(row["up"])
			dep_rc = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
			dep_em = np.load(os.path.join(em_base_path, f"EM-virtual-{proj_ident}-{em_idx}", f"{img_idx:06d}.npy"))
			rgb_rc = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"), cv2.IMREAD_GRAYSCALE)
			rgb_em = cv2.imread(os.path.join(em_base_path, f"EM-virtual-{proj_ident}-{em_idx}", f"{img_idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
			rgb_rc = rgb_rc[9 : 255, ...]
			rgb_rc = cv2.resize(rgb_rc, (224, 224))
			rgb_em = rgb_em / 255.0
			rgb_rc = rgb_rc / 255.0

			if show_fix:
				rgb_gt, dep_gt = get_depth_map(p, position, orientation, up, get_outputs = True)
				rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGBA2GRAY)
				rgb_gt = rgb_gt / 255.0
			
			
			"""
			if show_fix:
				out_dt = [(rgb_rc, "GSF-RC"), (rgb_em, "GSF-EM"), (rgb_gt, "GSF-GT"), (dep_rc, "DM-RC-SFS"), (dep_em, "DM-EM-Mesh"), (dep_gt, "DM-GT-Mesh")]
				num_plots = 2, 3
			else:
				out_dt = [(rgb_rc, "GSF-RC"), (rgb_em, "GSF-EM"), (dep_rc, "DM-RC-SFS"), (dep_em, "DM-EM-Mesh")]
				num_plots = 2, 2
			
			"""
			out_dt = [(rgb_rc, "GSF-RC"), (rgb_em, "GSF-VC-EM")]
			num_plots = 1, 2
			

			for i, (out_img, out_title) in enumerate(out_dt):
				plt.subplot(*num_plots, i + 1)
				if out_title.find("DM") != -1:
					plt.imshow(out_img)
				else:
					plt.imshow(out_img, cmap = "gray", vmin = 0, vmax = 1)
				if uses_colorbar:
					plt.colorbar()
				plt.axis("off")
				plt.title(out_title, fontsize = 20, y = -0.15)
			#plt.suptitle(f"REAL-{em_idx}-{img_idx}, {human_eval}", y = 0.05)

			#plt.suptitle(f"REAL-{em_idx}, Frame-{img_idx}")
			#plt.show()
			plt.savefig(os.path.join(out_path, f"{em_idx}-{img_idx:04d}.png"))


def sfs_distrib():
	out_path = "./depth-images/sfs-info"
	#out_path = "./depth-images/fix-noproj-GSFonly"
	chosen_data = [(0, 630), (0, 636)]
	os.makedirs(out_path, exist_ok = True)
	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in tqdm(reader):
			plt.clf()
			em_idx, img_idx, human_eval = int(row["em_idx"]), int(row["img_idx"]), int(row["human_eval"])
			if human_eval != 1:
				continue
			dep_rc = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
			#print(dep_rc.shape)
			#rgb_rc = rgb_rc[9 : 255, ...]
			#dep_rc = dep_rc[9 : 255, ...]
			plt.subplot(1, 2, 1)
			plt.imshow(dep_rc)
			plt.colorbar()
			plt.subplot(1, 2, 2)
			plt.hist(dep_rc.ravel(), bins = 20)
			plt.suptitle(f"REAL-{em_idx}-{img_idx}, {human_eval}")
			#plt.suptitle(f"REAL-{em_idx}, Frame-{img_idx}")
			#plt.show()
			plt.savefig(os.path.join(out_path, f"{em_idx}-{img_idx:04d}.png"))


def sfs_distrib_summ():
	out_path = "./depth-images/sfs-info-summ"
	os.makedirs(out_path, exist_ok = True)
	all_ranges, all_means, all_peaks = [], [], []
	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in tqdm(reader):
			em_idx, img_idx, human_eval = int(row["em_idx"]), int(row["img_idx"]), int(row["human_eval"])
			dep_rc = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy")).ravel()
			all_ranges.append(dep_rc.max() - dep_rc.min())
			all_means.append(dep_rc.mean())
			hst, __ = np.histogram(dep_rc, bins = 30)
			all_peaks.append(np.argmax(hst) + 1)

	np.save("all_ranges.npy", all_ranges)
	np.save("all_means", all_means)
	np.save("all_peaks", all_peaks)

	plt.figure(figsize = (12, 3.5))
	plt.subplots_adjust(left = 0.01, right = 0.99, top = 0.9, bottom = 0.3, hspace = 0.05, wspace = 0.15)
	#title_config = {"fontsize": 20, "y": -0.5}
	xlabel_config = {"fontsize": 20, "y": 1}
	ylabel_config = {"fontsize": 20}
	plt.subplot(1, 3, 1)
	plt.hist(all_ranges, bins = 30, density = True)
	#plt.title("Range Distribution", **title_config)
	plt.xlabel("DM-RC-SFS Range", **xlabel_config)
	plt.ylabel("Density", **ylabel_config)
	plt.subplot(1, 3, 2)
	plt.hist(all_means, bins = 30, density = True)
	#plt.title("Mean Distribution", **title_config)
	plt.xlabel("DM-RC-SFS Mean", **xlabel_config)
	plt.ylabel("Density", **ylabel_config)
	plt.subplot(1, 3, 3)
	plt.hist(all_peaks, bins = 30, range = (0, 30), density = True)
	#plt.title("Bin Peak Distribution", **title_config)
	plt.xlabel("Peak Bin of Histogram", **xlabel_config)
	plt.ylabel("Density", **ylabel_config)
	#plt.show()
	print(np.sum(np.array(all_peaks) <= 3) / len(all_peaks))


def plot_sfs_out_img():
	chosen_data = [(0, 36, 6.82), (0, 630, 5.29), (0, 636, 4.86)]
	out_path = "./depth-images/final-sfs-selected"
	os.makedirs(out_path, exist_ok = True)

	plt.figure(figsize = (12, 3.5))
	plt.subplots_adjust(left = 0.01, right = 0.99, top = 0.9, bottom = 0.3, hspace = 0.05, wspace = 0.15)
	title_config = {"fontsize": 20, "y": -0.5}

	for em_idx, img_idx, bir_thres in chosen_data:
		dep_rc = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
		rgb_rc = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"), cv2.IMREAD_GRAYSCALE)
		rgb_rc = rgb_rc[9 : 255, ...]
		rgb_rc = cv2.resize(rgb_rc, (224, 224))
		rgb_rc = rgb_rc / 255.0
		#print(rgb_rc.shape, dep_rc.shape)
		#print(dep_rc.min(), dep_rc.max())
		
		
		plt.subplot(1, 3, 1)
		plt.imshow(rgb_rc, cmap = "gray", vmin = 0, vmax = 1)
		plt.title("GSF-RC", **title_config)
		plt.axis("off")
		plt.subplot(1, 3, 2)
		plt.imshow(dep_rc, vmin = 2.5, vmax = 8)
		plt.title("DM-RC-SFS", **title_config)
		plt.colorbar()
		plt.axis("off")
		plt.subplot(1, 3, 3)
		plt.hist(dep_rc.ravel(), bins = 30, density = True, range = (2.5, 8))
		plt.title("DM Histogram", **title_config)
		plt.xlabel("Depth Value", fontsize = 15, y = 1)
		plt.ylabel("Density", fontsize = 15)
		plt.ylim(0, 1.0)
		
		plt.show()

		hst, bin_edges = np.histogram(dep_rc.ravel(), bins = 30, density = True)
		for i, x in enumerate(bin_edges):
			if bir_thres <= x:
				bir_idx = i - 1
				break
		print(np.max(hst[bir_idx : ]) / np.max(hst))
		print(np.sum(dep_rc >= bir_thres) / len(dep_rc.ravel()))
		plt.imshow(dep_rc >= bir_thres)
		plt.show()

		#plt.savefig(os.path.join(out_path, f"{em_idx}-{img_idx:04d}.png"))
		#plt.clf()


def plot_scaled_contours():
	chosen_data = [(0, 24), (0, 644), (0, 822), (0, 1058), (1, 1926), (1, 2404), (2, 1320)]
	out_path = "./depth-images/fix-final-multiscale-contours"
	os.makedirs(out_path, exist_ok = True)
	p = pv.Plotter(off_screen = True, window_size = (224, 224))
	p.add_mesh(pv.read("./meshes/Airway_Phantom_AdjustSmooth.stl"))

	plt.figure(figsize = (6, 3.5))
	plt.subplots_adjust(left = 0.01, right = 0.99, top = 0.99, bottom = 0.04, hspace = 0.05, wspace = 0.05)
	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in tqdm(reader):
			plt.clf()
			em_idx, img_idx, human_eval = int(row["em_idx"]), int(row["img_idx"]), int(row["human_eval"])
			"""
			if (em_idx, img_idx) not in chosen_data:
				continue
			"""
			if human_eval != 1:
				continue
			position, orientation, up = str_to_arr(row["position"]), str_to_arr(row["orientation"]), str_to_arr(row["up"])
			dep_rc = np.load(os.path.join(em_base_path, f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy"))
			rgb_rc = cv2.imread(os.path.join(em_base_path, f"EM-RGB-{em_idx}", f"{img_idx}.png"), cv2.IMREAD_GRAYSCALE)
			rgb_rc = rgb_rc[9 : 255, ...]
			rgb_rc = cv2.resize(rgb_rc, (224, 224))
			rgb_rc = cv2.cvtColor(rgb_rc, cv2.COLOR_GRAY2RGB)
			#rgb_rc = rgb_rc / 255.0

			rgb_gt, dep_gt = get_depth_map(p, position, orientation, up, get_outputs = True)
			rgb_gt = cv2.cvtColor(cv2.cvtColor(rgb_gt, cv2.COLOR_RGBA2GRAY), cv2.COLOR_GRAY2RGB)
			#rgb_gt = rgb_gt / 255.0
			
			scale_rates = [0.1, 0.25, 0.4]
			colour_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
			rc_contour_list, quantile_list, __ = cache_multiscale_base_data(dep_rc, scale_rates = scale_rates)
			for q, colour in zip(quantile_list, colour_list):
				rgb_rc = draw_contours(dep_rc, rgb_rc, q, colour)
				rgb_gt = draw_contours(dep_gt, rgb_gt, q, colour)

			out_dt = [(rgb_rc, "GSF-RC"), (rgb_gt, "GSF-VC-GT")]
			num_plots = 1, 2


			for i, (out_img, out_title) in enumerate(out_dt):
				plt.subplot(*num_plots, i + 1)
				plt.imshow(out_img)
				plt.axis("off")
				plt.title(out_title, fontsize = 20, y = -0.15)
				all_boxes = []
				for q, colour in zip(scale_rates, ["red", "green", "blue"]):
					colour_box = mpatches.Patch(color = colour, label = f"{q:.2f}")
					all_boxes.append(colour_box)
				plt.legend(handles = all_boxes, loc = "upper right", framealpha = 0.5, frameon = True)
			#plt.suptitle(f"REAL-{em_idx}-{img_idx}, {human_eval}", y = 0.05)

			#plt.suptitle(f"REAL-{em_idx}, Frame-{img_idx}")
			#plt.show()
			plt.savefig(os.path.join(out_path, f"{em_idx}-{img_idx:04d}.png"))



if __name__ == '__main__':
	#plt.figure(figsize = (20, 15))
	#write_sim_list()
	#plot_sim_list()
	#plot_corr_mega()
	#plot_bf_fix(0, 8)
	#find_correspond_tryidx()
	#plot_fix_maxfit([0, 1, 2])
	#select_between_candidates(2, [2, 3, 4, 8, 9], 100)
	#plot_max_depth()
	"""
	if sys.argv[1] == "eval":
		plot_bf_fix(ing(sys.argv[2]), int(sys.argv[3]))
	if sys.argv[1] == "select":
		manually_select_alignment(int(sys.argv[2]), int(sys.argv[3]))
	"""
	
	#draw_contour_outputs(1, 0)
	"""
	draw_contour_outputs(0, 2)
	draw_contour_outputs(1, 2)
	draw_contour_outputs(2, 0)
	draw_contour_outputs(2, 2)
	"""


	"""

	manually_select_alignment_contour(0, 3)
	manually_select_alignment_contour(1, 3)
	manually_select_alignment_contour(2, 3)
	"""
	"""
	draw_interp_outputs(0, 4)
	draw_interp_outputs(1, 4)
	draw_interp_outputs(2, 4)
	"""

	"""
	manually_select_alignment_interp(0, 4)
	manually_select_alignment_interp(1, 4)
	manually_select_alignment_interp(2, 4)
	"""

	"""
	manually_select_alignment_contour(0, 0)
	manually_select_alignment_contour(0, 2)
	manually_select_alignment_contour(1, 0)
	manually_select_alignment_contour(1, 2)
	manually_select_alignment_contour(2, 0)
	manually_select_alignment_contour(2, 2)
	"""

	#draw_contour_outputs(2, 0)

	#plot_sim_scatter()
	#plot_final_fixes(True)
	#plot_out_img(proj = False, show_fix = False)
	#sfs_distrib()
	#plot_sfs_out_img()
	#sfs_distrib_summ()
	plot_scaled_contours()
