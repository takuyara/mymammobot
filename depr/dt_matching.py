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

def load_image(path, target_shape):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	tensor = K.image_to_tensor(img, None).float() / 255.0
	tensor = KG.transform.resize(tensor, target_shape)
	tensor = KG.transform.center_crop(tensor, (target_shape, target_shape))
	return tensor

def write_image(tensor):
	return K.tensor_to_image((tensor * 255.0).byte())

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

def select_ct_candidates(em_img_path, ct_points, ct_paths, candidate_distance, maximum_candidates):
	em_pose = np.loadtxt(em_img_path.replace(".png", ".txt")).reshape(-1)[ : 3]
	all_dists = np.sum((ct_points - em_pose) ** 2, axis = 1) ** 0.5
	em_ct_pair = []
	for i in np.argsort(all_dists)[ : maximum_candidates]:
		if all_dists[i] > candidate_distance:
			break
		em_ct_pair.append((em_img_path, ct_paths[i]))
	return em_ct_pair

def compute_all_coords(em_ct_pair, em_cand_cnt, batch_size, target_shape, lr, num_iters, tolerance, pyramid_levels, transform_type, device):
	em_batch, ct_batch, em_batch_path, ct_batch_path = [], [], [], []
	em_cand_done, em_best_reg = {}, {}
	csv_mtdt = [["em_path", "ct_path", "correlation"]]
	for em_path, ct_path in em_ct_pair:
		em_img, ct_img = load_image(em_path, target_shape).to(device), load_image(ct_path, target_shape).to(device)
		em_batch.append(em_img)
		ct_batch.append(ct_img)
		em_batch_path.append(em_path)
		ct_batch_path.append(ct_path)
		if len(em_batch) == batch_size:
			registrator = ImageRegistrator(batch_size, target_shape, target_shape, num_iters = num_iters, tolerance = tolerance,
				lr = lr, pyramid_levels = pyramid_levels, transform_type = transform_type).to(device)
			out = registrator(torch.cat(em_batch, dim = 0), torch.cat(ct_batch, dim = 0))
			for em_path, ct_path, orig_em_img, reg_em_img in zip(em_batch_path, ct_batch_path, em_batch, torch.unbind(out)):
				this_corr = img_corr(reg_em_img, orig_em_img)
				if em_path not in em_cand_done:
					em_cand_done[em_path] = 1
					em_best_reg[em_path] = (this_corr, ct_path, reg_em_img.detach().cpu())
				else:
					em_cand_done[em_path] += 1
					if em_best_reg[em_path][0] < this_corr:
						em_best_reg[em_path] = (this_corr, ct_path, reg_em_img.detach().cpu())
				if em_cand_done[em_path] == em_cand_cnt[em_path]:
					this_corr, ct_path, reg_em_img = em_best_reg[em_path]
					csv_mtdt.append([em_path, ct_path, this_corr])
					cv2.imwrite(em_path.replace(".png", "_reg.png"), write_image(reg_em_img))

					plt.subplot(1, 3, 1)
					plt.imshow(write_image(load_image(ct_path, target_shape)), cmap = "gray")
					plt.title("CT")
					plt.subplot(1, 3, 2)
					plt.imshow(write_image(load_image(em_path, target_shape)), cmap = "gray")
					plt.title("EM")
					plt.subplot(1, 3, 3)
					plt.imshow(write_image(reg_em_img), cmap = "gray")
					plt.title("EM (transformed to CT)")
					plt.show()
			
			em_batch, ct_batch, em_batch_path, ct_batch_path = [], [], [], []
	with open("registration_metadata.csv", "w", newline = "") as f:
		writer = csv.writer(f)
		writer.writerows(csv_mtdt)

def main():
	all_em_ct_pairs = []
	em_cand_cnt = {}
	ct_points, ct_paths = build_ct_pose_array(base_path)
	for i in range(len(os.listdir(em_all_path)) // 2):
		em_path = os.path.join(em_all_path, f"{i:06d}.png")
		if not os.path.exists(em_path):
			break
		this_pairs = select_ct_candidates(em_path, ct_points, ct_paths, candidate_distance, maximum_candidates)
		all_em_ct_pairs.extend(this_pairs)
		em_cand_cnt[em_path] = len(this_pairs)
		#print(em_path, len(this_pairs))
	compute_all_coords(all_em_ct_pairs, em_cand_cnt, batch_size, target_shape, lr, num_iters, tolerance, pyramid_levels, transform_type, device)

if __name__ == '__main__':
	main()


ct_img, em_img = load_image(path_ct, target_shape).to(device), load_image(path_em, target_shape).to(device)
ct_img, em_img = ct_img.repeat(batch_size, 1, 1, 1), em_img.repeat(batch_size, 1, 1, 1)
#registrator = KG.ImageRegistrator("similarity", loss_fn = corr_loss, num_iterations = 1000, pyramid_levels = pyramid_levels)
#registrator = KG.ImageRegistrator("similarity")
registrator = ImageRegistrator(batch_size, target_shape, target_shape, num_iters = 200, tolerance = 1e-8, lr = 1e-3).to(device)
out, losses = registrator(em_img, ct_img, return_loss = True)


#print(nn.L1Loss()(mov, out), nn.L1Loss()(ref, out1), nn.L1Loss()(ref, mov))

#print(losses, len(losses))

#plt.plot(range(len(losses)), losses)

plt.plot(losses)
plt.xlabel("# Iteration")
plt.ylabel("Negative Correlation")
plt.title("Negative Correlation Loss Curve for Sample Registration")
plt.show()




