import matplotlib.pyplot as plt
import cv2
import numpy as np
import kornia as K
import kornia.geometry as KG
import torch
from torch import nn
from image_registration import ImageRegistrator, img_corr
import csv
import os
from tqdm import tqdm

base_path = "D:\\hope\\mymammobot\\depth-images"
em_all_path = "D:\\hope\\mymammobot\\depth-images\\EM-0"
target_shape = 224
candidate_distance = 5
maximum_candidates = 1
pyramid_levels = 3
lr = 1e-4
tolerance = 1e-6
num_iters = 2000
batch_size = 4
device = "cuda"
transform_type = "rigid"

def load_n_reshape(path, target_shape):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	crop_size = min(img.shape[0], img.shape[1])
	x = img.shape[0] / 2 - crop_size / 2
	y = img.shape[1] / 2 - crop_size / 2
	img = img[int(x) : int(x + crop_size), int(y) : int(y + crop_size)]
	img = cv2.resize(img, (target_shape, target_shape))
	return img

def build_ct_pose_array(base_path):
	all_points, all_paths = [], []
	for this_ct in os.listdir(base_path):
		if this_ct.startswith("CL"):
			this_ct = os.path.join(base_path, this_ct)
			for i in range(len(os.listdir(this_ct)) // 2):
				all_points.append(np.loadtxt(os.path.join(this_ct, f"{i:06d}.txt")).reshape(-1)[ : 3])
				all_paths.append(os.path.join(this_ct, f"{i:06d}.png"))
	all_points = np.stack(all_points, axis = 0)
	return all_points, all_paths

def inc_dict(dct, k, v):
	if k not in dct:
		dct[k] = 0
	dct[k] += v

def select_ct_candidates(em_img_path, ct_points, ct_paths, candidate_distance, em_weights, ct_weights):
	em_pose = np.loadtxt(em_img_path.replace(".png", ".txt")).reshape(-1)[ : 3]
	all_dists = np.sum((ct_points - em_pose) ** 2, axis = 1) ** 0.5
	indices = np.where(all_dists < candidate_distance)[0]
	if len(indices) > 0:
		inc_dict(em_weights, em_img_path, 1)
		for i in indices:
			inc_dict(ct_weights, ct_paths[i], 1 / len(indices))

def compute_intensity_hist(weight_dict, target_shape):
	hist = np.zeros(256)
	for path, t_weight in tqdm(weight_dict.items()):
		img = load_n_reshape(path, target_shape)
		intensities, counts = np.unique(img, return_counts = True)
		hist[intensities] += t_weight * counts
	hist = hist / np.sum(hist)
	return hist

def main():
	ct_points, ct_paths = build_ct_pose_array(base_path)
	em_weights, ct_weights = {}, {}
	print("Matching EM with CT")
	for this_em in os.listdir(base_path):
		if this_em.startswith("EM"):
			this_em = os.path.join(base_path, this_em)
			for i in tqdm(range(len(os.listdir(this_em)) // 2)):
				em_path = os.path.join(this_em, f"{i:06d}.png")
				if not os.path.exists(em_path):
					break
				select_ct_candidates(em_path, ct_points, ct_paths, candidate_distance, em_weights, ct_weights)
	print("Computing histogram")
	em_hist = compute_intensity_hist(em_weights, target_shape)
	ct_hist = compute_intensity_hist(ct_weights, target_shape)
	np.save("em_hist.npy", em_hist)
	np.save("ct_hist.npy", ct_hist)
	plt.subplot(1, 2, 1)
	plt.bar(range(256), em_hist)
	plt.title("EM Intensity Histogram")
	plt.subplot(1, 2, 2)
	plt.bar(range(256), ct_hist)
	plt.title("CT Intensity Histogram")
	plt.show()

if __name__ == '__main__':
	main()



