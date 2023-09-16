import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import directed_hausdorff as DHD
import os
from tqdm import tqdm
import cv2
import csv
import argparse
import time
from multiprocessing import Pool

from ds_gen.camera_features import camera_params
from utils.cl_utils import project_to_cl, load_all_cls, in_mesh_bounds, get_cl_direction
from utils.geometry import random_points_in_sphere, arbitrary_perpendicular_vector, rotate_single_vector, rotate_all_degrees, get_vector_angle, random_perpendicular_offsets
from pose_fixing.similarity import comb_corr_sim, reg_mse_sim, mi_sim, corr_sim
from ds_gen.depth_map_generation import get_depth_map
from pose_fixing.move_camera import randomise_params
from utils.stats import weighted_corr
#from domain_transfer.alignment import reg_depth_maps

default_params = {
	"coarse": {
		"position_scale": 7,
		"position_samples": 30,
		"orientation_scale": 0.3,
		"orientation_samples": 30,
		"up_samples": 10,
	},
	"fine": {
		"position_scale": 2,
		"position_samples": 15,
		"orientation_scale": 0.1,
		"orientation_samples": 15,
		"up_samples": 30,
	},
}

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--em-base-path", type = str, default = "./depth-images")
	parser.add_argument("--cl-base-path", type = str, default = "./CL")
	parser.add_argument("--output-metadata", type = str, default = "register_params.csv")
	parser.add_argument("--filter-oob", action = "store_true", default = False)
	parser.add_argument("--angle-threshold", type = float, default = 20)
	parser.add_argument("--em-idx", type = int, default = 0)
	parser.add_argument("--try-idx", type = int, default = 0)
	parser.add_argument("--init-idx", type = int, default = 0)
	parser.add_argument("--pool-size", type = int, default = 7)
	parser.add_argument("--step-size", type = int, default = 8)
	parser.add_argument("--window-size", type = int, default = 224)
	parser.add_argument("--focal-scale", type = float, default = 0)
	parser.add_argument("--position-scale", type = float, default = 7)
	parser.add_argument("--orientation-scale", type = float, default = 0.3)
	parser.add_argument("--focal-samples", type = int, default = 1)
	parser.add_argument("--position-samples", type = int, default = 30)
	parser.add_argument("--orientation-samples", type = int, default = 30)
	parser.add_argument("--up-samples", type = int, default = 10)
	parser.add_argument("--uses_new_rotation", action = "store_true", default = False)
	return parser.parse_args()

def get_fixed_corr(ref_depth_map, p, focal_length, camera_position, camera_orientation, up_direction):
	depth_map = get_depth_map(p, camera_position, camera_orientation, up_direction, focal_length = focal_length)
	corr = comb_corr_sim(ref_depth_map, depth_map)
	return corr, focal_length, camera_position, camera_orientation, up_direction

def check_rotation(frame_idx, em_path, em_depth_path, output_path, args):
	surface = pv.read(args.mesh_path)
	p = pv.Plotter(off_screen = True, window_size = (args.window_size, args.window_size))
	p.add_mesh(surface)


	real_depth_map = np.load(os.path.join(em_depth_path, f"{frame_idx:06d}.npy"))
	real_rgb = cv2.imread(os.path.join(args.em_base_path, f"EM-RGB-{args.em_idx}", f"{frame_idx}.png"))
	#print(os.path.join(em_depth_path, f"{frame_idx:06d}.npy"), os.path.join(em_path, f"{frame_idx:06d}.png"))

	print("In check rotation")
	num_samples = 500
	up_samples = 180

	"""
	optim_position = np.array([-32.71902941, -25.05143104, -163.36083433])
	optim_orientation = np.array([-0.87824484, -0.41870292, -0.23102786])
	optim_up = np.array([ 0.47559157, -0.71424555, -0.51348414])
	"""
	#0.8872

	"""
	max_optim = 0.8872
	for this_degree in np.arange(-5, 5, 0.2):
		t_up = rotate_single_vector(optim_up, optim_orientation, this_degree)
		virtual_depth_map = get_depth_map(p, camera_params["focal_length"], optim_position, optim_orientation, t_up)
		this_corr = comb_corr_sim(real_depth_map, virtual_depth_map)
		if this_corr > max_optim:
			max_optim, best_up = this_corr, t_up

	rgb, virtual_depth_map = get_depth_map(p, camera_params["focal_length"], optim_position, optim_orientation, optim_up, get_outputs = True)
	
	plt.clf()
	plt.subplot(2, 2, 1)
	plt.imshow(real_depth_map, cmap = "gray")
	plt.subplot(2, 2, 2)
	plt.imshow(virtual_depth_map, cmap = "gray")
	plt.subplot(2, 2, 3)
	plt.imshow(rgb)
	plt.subplot(2, 2, 4)
	plt.scatter(real_depth_map.flatten(), virtual_depth_map.flatten())
	plt.savefig(os.path.join(output_path, f"{frame_idx:06d}-{max_optim:.4f}.png"))
	with open(os.path.join(output_path, f"{frame_idx:06d}-{max_optim:.4f}.txt"), "w") as f:
		out = f"{optim_position}\n{optim_orientation}\n{best_up}\n"
		f.write(out)
	plt.show()



	print("Best corr: ", max_optim, " up: ", t_up)
	exit()
	"""

	# Show error in weighted correlation


	#Iter 0
	"""
	View angle = 120, counter example for weights
	old_position = np.array([-33.79384209, -28.94875791, -160.28407521])
	old_orientation = np.array([-0.89581517, 0.32385322, -0.30435879])
	old_up = np.array([0.43895517, 0.53761082, -0.71992567])
	#desired_up = rotate_single_vector(old_up, t_orientation, 270)

	new_position = np.array([-32.71902941, -25.05143104, -163.36083433])
	new_orientation = np.array([-0.87824484, -0.41870292, -0.23102786])
	new_up = np.array([0.46974083, -0.66481073, -0.58083582])
	"""


	
	old_position = np.array([-43.13930405, -29.77635236, -160.71365191])
	old_orientation = np.array([-0.93962246, -0.15040396, -0.30738945])
	old_up = np.array([0.32958453, -0.15596661, -0.93115437])
	"""
	old_position = np.array([-31.68429595, -27.44295559, -161.55171964])
	old_orientation = np.array([-1.01959425, 0.13263613, -0.18120493])
	old_up = np.array([-0.17030983, 0.58201775, -0.79514143])
	"""

	"""
	Best
	new_position = np.array([-35.23436981, -26.07758012, -161.09738046])
	new_orientation = np.array([-0.86110491, -0.40747873, -0.30407139])
	new_up = np.array([0.50811759, -0.71058364, -0.4867108])
	"""
	

	
	#Iter 1
	new_position = np.array([-33.013039, -27.23421861, -162.06400843])
	new_orientation = np.array([-0.91553654, -0.17466719, -0.36233164])
	new_up = np.array([0.34834557, -0.79469893, -0.49710057])
	

	
	#Iter 2
	"""
	t_position = np.array([-32.74093312, -24.63603652, -161.3075079])
	t_orientation = np.array([-0.74968252, -0.45564735, -0.47996001])
	desired_up = np.array([0.66018554, -0.56548868, -0.49434564])
	"""
	

	"""
	Iter 3
	t_position = np.array([-30.59947988, -24.11401394, -160.20225996])
	t_orientation = np.array([-0.72133924, -0.39416521, -0.56947651])
	desired_up = np.array([0.69216001, -0.43897778, -0.57289879])
	"""

	"""
	virtual_depth_map = get_depth_map(p, camera_params["focal_length"], t_position, t_orientation, desired_up)
	plt.imshow(virtual_depth_map)
	plt.show()

	lipu_up = np.array([0.47460083, 0.75710684, -0.44893572])

	virtual_depth_map = get_depth_map(p, camera_params["focal_length"], t_position, t_orientation, lipu_up)
	plt.imshow(virtual_depth_map)
	plt.show()
	print(get_vector_angle(lipu_up, desired_up))
	"""
	"""
	t_up = rotate_single_vector(up_vector_base, t_orientation, 270)
	t_left = np.cross(t_up, t_orientation)
	t_left = t_left / np.linalg.norm(t_left)
	"""

	"""
	t_position = new_position
	t_orientation = new_orientation
	desired_up = new_up

	axial_scale = 5
	radial_scale = 5
	orientation_scale = 0.5

	all_sampled_params = randomise_params(camera_params["focal_length"],
		axial_scale, radial_scale, t_position, orientation_scale, t_orientation,
		num_samples, up_samples)
	"""
	
	#best_corr_params = get_fixed_corr(real_depth_map, p, camera_params["focal_length"], t_position, t_orientation, up_vector_base)

	#plt.figure(figsize = (20, 15))

	"""
	left_offset = -2
	up_offset = 3
	axial_offset = 4
	up_rotate = -7
	ori_left_offset = 0
	ori_up_offset = 0.3
	new_position = t_position + left_offset * t_left + up_offset * t_up + axial_offset * t_orientation
	new_up = rotate_single_vector(t_up, t_orientation, up_rotate)
	new_orientation = t_orientation + t_up * ori_up_offset + t_left * ori_left_offset
	"""

	"""
	new_rgb, new_dep = get_depth_map(p, camera_params["focal_length"], new_position, new_orientation, new_up, get_outputs = True)
	plt.subplot(2, 2, 1)
	plt.imshow(new_dep)
	plt.subplot(2, 2, 2)
	plt.imshow(new_rgb)
	plt.subplot(2, 2, 3)
	plt.imshow(real_depth_map)
	plt.subplot(2, 2, 4)
	plt.imshow(real_rgb)
	plt.show()
	"""

	from sklearn.linear_model import LinearRegression as LR
	from pose_fixing.similarity import dark_threshold_r, dark_threshold_v, dark_weight, light_weight

	light_additional_weight = 1.2

	def weighted_mean(x, w):
		#print(x.shape, w.shape)
		return (x * w).sum() / w.sum()

	def weighted_cov(x, y, w):
		return (w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))).sum() / w.sum()

	def get_threshold(x, bins = 20, rate = 0.1):
		#return np.median(x.flatten()) * 2 - x.min()
		h, bin_edges = np.histogram(x.flatten(), bins = bins, density = True)
		first_outlier_idx = len(h) - np.argmax(np.flip((h > np.max(h) * rate).astype(int)))
		return bin_edges[first_outlier_idx]

	def get_old_weights(rd1, vd1):
		r_thres = get_threshold(rd1)
		v_thres = get_threshold(vd1)
		dark_mask = np.logical_and(rd1 < r_thres, vd1 < v_thres)
		w = np.ones_like(rd1) * light_weight
		w[dark_mask] = dark_weight
		return w

	def get_new_weights(rd1, vd1):
		r_thres = get_threshold(rd1)
		v_thres = get_threshold(vd1)
		dark_mask = np.logical_and(rd1 < r_thres, vd1 < v_thres)
		dark_count = np.sum(dark_mask)
		light_count = len(rd1) - dark_count
		w = np.ones_like(rd1) * light_additional_weight * dark_count / (dark_count + light_count)
		w[dark_mask] = light_count / (dark_count + light_count)
		return w

	def plot_it(rgb, virtual_depth_map, error_mode = "corr", weights = "new", mean_method = "weighted"):
		rd1, vd1 = real_depth_map.flatten(), virtual_depth_map.flatten()
		w = get_old_weights(rd1, vd1) if weights == "old" else get_new_weights(rd1, vd1)
		if mean_method == "weighted":
			err = ((rd1 - weighted_mean(rd1, w)) * (vd1 - weighted_mean(vd1, w))) / (weighted_cov(rd1, rd1, w) * weighted_cov(vd1, vd1, w)) ** 0.5
		else:
			err = ((rd1 - np.mean(rd1)) * (vd1 - np.mean(vd1))) / (np.var(rd1) * np.var(vd1))
		corr = np.sum(err * w) / np.sum(w)
		err = err.reshape(*virtual_depth_map.shape)
		w = w.reshape(*virtual_depth_map.shape)

		return err, w, corr

	def analysis_corr(rd1, vd1):
		prev_shape = rd1.shape
		rd1, vd1 = rd1.flatten(), vd1.flatten()
		r_thres = get_threshold(rd1)
		q = np.sum(rd1 < r_thres) / len(rd1)
		#v_thres = get_threshold(vd1)
		v_thres = np.quantile(vd1, q)
		dark_mask = np.logical_and(rd1 < r_thres, vd1 < v_thres)
		w = np.ones_like(rd1) * light_weight
		w[dark_mask] = dark_weight
		err = ((rd1 - weighted_mean(rd1, w)) * (vd1 - weighted_mean(vd1, w))) / (weighted_cov(rd1, rd1, w) * weighted_cov(vd1, vd1, w)) ** 0.5
		print(f"Dark count rate {(np.sum(dark_mask) / len(rd1)):.4f}. Dark weight rate {(np.sum(w[dark_mask]) / np.sum(w)):.4f}.")
		print(f"Dark cov unweighted contribute: {np.mean(err[dark_mask]):.4f}.")
		print(f"Light count rate {(1 - np.sum(dark_mask) / len(rd1)):.4f}. Light weight rate {(np.sum(w[np.logical_not(dark_mask)]) / np.sum(w)):.4f}")
		print(f"Light cov unweighted contribute: {np.mean(err[np.logical_not(dark_mask)]):.4f}.")
		light_mask = np.logical_and
		print(f"IoU: {np.sum(np.logical_and(rd1 > r_thres, vd1 > v_thres)) / np.sum(np.logical_or(rd1 > r_thres, vd1 > v_thres)):.4f}")
		print(f"Quantile: {q:.4f}. Correlation: {(np.sum(err * w) / np.sum(w)):.4f}.")
		#print(f"Unweighted Correlation: {}")

		light_mask_r = (rd1 > r_thres).astype(np.uint8)
		light_mask_v = (vd1 > v_thres).astype(np.uint8)
		light_contours_r, __ = cv2.findContours(light_mask_r.reshape(*prev_shape), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		light_contours_v, __ = cv2.findContours(light_mask_v.reshape(*prev_shape), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		contours_img_r = cv2.drawContours(cv2.cvtColor(light_mask_r.reshape(*prev_shape), cv2.COLOR_GRAY2BGR), light_contours_r, -1, (0, 255, 0), 3)
		contours_img_v = cv2.drawContours(cv2.cvtColor(light_mask_v.reshape(*prev_shape), cv2.COLOR_GRAY2BGR), light_contours_v, -1, (0, 255, 0), 3)

		"""
		print(light_contours_r.shape, light_contours_v.shape)
		print(len(light_contours_r), len(light_contours_v))


		plt.subplot(1, 2, 1)
		plt.imshow(contours_img_r)
		plt.title("Real")
		plt.subplot(1, 2, 2)
		plt.imshow(contours_img_v)
		plt.title("Virtual")
		plt.show()
		"""

		light_contours_v = np.concatenate(light_contours_v, axis = 0).reshape(-1, 2)
		light_contours_r = np.concatenate(light_contours_r, axis = 0).reshape(-1, 2)
		print(light_contours_r.shape, light_contours_v.shape)
		hausdorff_dist = max(DHD(light_contours_r, light_contours_v)[0], DHD(light_contours_v, light_contours_r)[0])
		print(f"Hausdorff distance: {hausdorff_dist:.4f}.")

		iou = np.zeros_like(rd1)
		iou[np.logical_and(rd1 > r_thres, vd1 > v_thres)] = 1
		iou[np.logical_and(rd1 > r_thres, vd1 <= v_thres)] = 2
		iou[np.logical_and(rd1 <= r_thres, vd1 > v_thres)] = 3

		iou_value = np.sum(np.logical_and(rd1 > r_thres, vd1 > v_thres)) / np.sum(np.logical_or(rd1 > r_thres, vd1 > v_thres))
		#return iou.reshape(*prev_shape)
		return iou_value, hausdorff_dist, contours_img_r, contours_img_v

		#print(f"Confirmed corr {(np.sum(err * w) / np.sum(w)):.4f}")

	def log_error(x1, x2):
		alp = np.mean(np.log(x1) - np.log(x2))
		dist = np.mean((np.log(x2) - np.log(x1) + alp) ** 2) / 2
		return dist

	old_rgb, old_dep = get_depth_map(p, old_position, old_orientation, old_up, get_outputs = True)
	err_old, mask_old, corr_old = plot_it(old_rgb, old_dep)
	#err_old, mask_old, corr_old = plot_it(old_rgb, old_dep, mean_method = "simple")
	new_rgb, new_dep = get_depth_map(p, new_position, new_orientation, new_up, get_outputs = True)
	err_new, mask_new, corr_new = plot_it(new_rgb, new_dep)
	#err_new, mask_new, corr_new = plot_it(new_rgb, new_dep, mean_method = "simple")

	"""
	plt.imshow(new_dep, cmap = "gray")
	plt.show()
	exit()
	"""



	print("Old sim (corr, log): ", corr_sim(real_depth_map, old_dep), comb_corr_sim(real_depth_map, old_dep), mi_sim(real_depth_map, old_dep, num_bins = 30), log_error(real_depth_map, old_dep))
	old_iou, old_hd, contours_img_r_old, contours_img_v_old = analysis_corr(real_depth_map, old_dep)
	print("New sim (corr, log): ", corr_sim(real_depth_map, new_dep), comb_corr_sim(real_depth_map, new_dep), mi_sim(real_depth_map, new_dep, num_bins = 30), log_error(real_depth_map, new_dep))
	new_iou, new_hd, contours_img_r_new, contours_img_v_new = analysis_corr(real_depth_map, new_dep)


	"""
	for fl in [20, 50, 100, 200, 300]:
		tmp_dep = get_depth_map(p, new_position, new_orientation, new_up, focal_length = fl)
		print(np.mean((tmp_dep - new_dep) ** 2))
	
	exit()
	"""

	"""
	bins = 20
	rate = 0.1

	plt.subplot(1, 3, 1)
	plt.hist(old_dep.flatten(), bins = bins, density = True)
	plt.title(f"Old depth histogram, threshold: {get_threshold(old_dep, bins, rate):.2f}")

	plt.subplot(1, 3, 2)
	plt.hist(new_dep.flatten(), bins = bins, density = True)
	plt.title(f"New depth histogram, threshold: {get_threshold(new_dep, bins, rate):.2f}")
	plt.subplot(1, 3, 3)
	plt.hist(real_depth_map.flatten(), bins = bins, density = True)
	plt.title(f"Real depth histogram, threshold: {get_threshold(real_depth_map, bins, rate):.2f}")
	plt.show()
	exit()
	"""

	plt.subplot(2, 2, 1)
	plt.imshow(real_depth_map, cmap = "gray")
	plt.title("Real SFS Depth")
	plt.subplot(2, 2, 2)
	plt.imshow(real_rgb)
	plt.title("Real RGB")
	#plt.imshow(err_old, vmin = -0.5, vmax = 5)
	#plt.hist(err1.flatten())
	plt.subplot(2, 2, 3)
	plt.imshow(old_dep, cmap = "gray")
	plt.title(f"Correlation: {comb_corr_sim(real_depth_map, old_dep):.4f}. IoU: {old_iou:.4f}. HD: {old_hd:.2f}")
	plt.subplot(2, 2, 4)
	plt.imshow(new_dep, cmap = "gray")
	plt.title(f"Correlation: {comb_corr_sim(real_depth_map, new_dep):.4f}. IoU: {new_iou:.4f}. HD: {new_hd:.2f}")
	#plt.imshow(old_dep, cmap = "gray", vmin = 0, vmax = 40)
	plt.show()
	exit()
	plt.subplot(2, 3, 5)
	#plt.imshow(new_dep, cmap = "gray", vmin = 0, vmax = 40)

	plt.colorbar()
	plt.title(f"New weighted mean: {weighted_mean(new_dep, mask_new):.4f}.")
	plt.subplot(2, 3, 6)
	#plt.imshow(mask_new)
	plt.imshow(new_iou)
	plt.colorbar()
	plt.show()

	plt.subplot(2, 3, 1)
	#plt.imshow(err_old, vmin = -0.5, vmax = 5)
	plt.imshow(err_old, cmap = "gray", vmin = 0, vmax = 5)
	plt.colorbar()
	plt.title(f"Old pose corr: {comb_corr_sim(real_depth_map, old_dep):.4f}")
	#plt.hist(err1.flatten())
	plt.subplot(2, 3, 2)
	plt.imshow(old_dep, cmap = "gray")
	plt.colorbar()
	plt.title(f"Old weighted mean: {weighted_mean(old_dep, mask_old):.4f}.")
	plt.subplot(2, 3, 3)
	#plt.imshow(mask_old)
	plt.imshow(old_iou)
	plt.colorbar()
	plt.subplot(2, 3, 4)
	#plt.hist(err2.flatten())
	#plt.imshow(err2, vmin = -0.5, vmax = 5)
	plt.imshow(err_new, cmap = "gray", vmin = 0, vmax = 5)
	plt.colorbar()
	plt.title(f"New pose corr: {comb_corr_sim(real_depth_map, new_dep):.4f}")
	#plt.imshow(old_dep, cmap = "gray", vmin = 0, vmax = 40)
	plt.subplot(2, 3, 5)
	#plt.imshow(new_dep, cmap = "gray", vmin = 0, vmax = 40)
	plt.imshow(new_dep, cmap = "gray")
	plt.colorbar()
	plt.title(f"New weighted mean: {weighted_mean(new_dep, mask_new):.4f}.")
	plt.subplot(2, 3, 6)
	#plt.imshow(mask_new)
	plt.imshow(new_iou)
	plt.colorbar()
	plt.show()

	exit()

	"""

	for i, (t_focal, t_position, t_orientation, t_up) in enumerate(tqdm(all_sampled_params)):
		#p1.add_mesh(pv.Arrow(t_position, t_orientation), color = "red")
		#print(get_vector_angle(t_up, desired_up))
		if get_vector_angle(t_up, desired_up) > 30:
			continue
		virtual_depth_map = get_depth_map(p, camera_params["focal_length"], t_position, t_orientation, t_up)
		if np.any(np.isnan(virtual_depth_map)):
			continue
		this_corr = comb_corr_sim(real_depth_map, virtual_depth_map)
		#cv2.imwrite(os.path.join(output_path, f"{frame_idx:06d}-{this_corr:.4f}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
		if this_corr > 0.85:
			plt.clf()
			plt.subplot(2, 2, 1)
			plt.imshow(real_depth_map, cmap = "gray")
			plt.subplot(2, 2, 2)
			plt.imshow(virtual_depth_map, cmap = "gray")
			plt.subplot(2, 2, 3)
			rgb, __ = get_depth_map(p, camera_params["focal_length"], t_position, t_orientation, t_up, get_outputs = True)
			plt.imshow(rgb)
			plt.subplot(2, 2, 4)
			plt.scatter(real_depth_map.flatten(), virtual_depth_map.flatten())
			plt.savefig(os.path.join(output_path, f"{frame_idx:06d}-{this_corr:.4f}.png"))
			with open(os.path.join(output_path, f"{frame_idx:06d}-{this_corr:.4f}.txt"), "w") as f:
				out = f"{t_position}\n{t_orientation}\n{t_up}\n"
				f.write(out)
	"""

def check_rotation_1(frame_idx, em_path, em_depth_path, output_path, args):
	real_depth_map = np.load(os.path.join(em_depth_path, f"{frame_idx:06d}.npy"))
	plt.subplot(1, 2, 1)
	plt.hist(real_depth_map.ravel(), bins = 20, density = True)
	plt.subplot(1, 2, 2)
	plt.imshow(real_depth_map, cmap = "gray")
	plt.show()

def main():
	args = get_args()
	
	args.em_idx = 1
	frame_idx = 1416
	"""
	args.em_idx = 0
	frame_idx = 680
	"""
	args.try_idx = "non-specific-tryidx"
	output_path = os.path.join(args.em_base_path, f"EM-virtual-autofix-{args.em_idx}-{frame_idx}-{args.try_idx}")
	em_path = os.path.join(args.em_base_path, f"EM-{args.em_idx}")
	em_depth_path = os.path.join(args.em_base_path, f"EM-rawdep-{args.em_idx}")
	os.makedirs(output_path, exist_ok = True)

	"""
	focal_length = 10
	translation = np.array([-68.30139041, -36.53458994, -163.48089302])
	orientation = np.array([-0.9871122, 0.12990399, -0.09345835])
	up_direction = np.array([-0.0494519, -0.80303439, -0.59387732])
	surface = pv.read(args.mesh_path)
	p = pv.Plotter(off_screen = True, window_size = (args.window_size, args.window_size))
	p.add_mesh(surface)
	rgb_img, __ = get_depth_map(p, focal_length, translation, orientation, up_direction, get_outputs = True)
	plt.imshow(rgb_img)
	plt.show()

	for frame_idx in range(20):
		check_rotation(frame_idx, em_path, em_depth_path, output_path, args)
	"""

	print("Before check rotation")
	check_rotation(frame_idx, em_path, em_depth_path, output_path, args)


if __name__ == '__main__':
	main()
