import os
import csv
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from domain_transfer.transformation import get_simple_loaders, SFS2Mesh, Mesh2SFS
from utils.preprocess import get_img_transform

from torchvision import transforms

def plot_one_batch(model, test_dloader, device):
	for sfs_imgs, mesh_imgs in test_dloader:
		with torch.no_grad():
			transformed = model(sfs_imgs.to(device).float(), mesh_imgs.to(device).float())[1]
		for i in range(min(len(sfs_imgs), 5)):
			plt.subplot(1, 3, 1)
			plt.imshow(sfs_imgs[i, ...].cpu().numpy(), cmap = "gray")
			plt.title("Original SFS")
			plt.colorbar()
			plt.subplot(1, 3, 2)
			plt.imshow(transformed[i, ...].cpu().numpy(), cmap = "gray")
			plt.title(f"Transformed")
			plt.colorbar()
			plt.subplot(1, 3, 3)
			plt.imshow(mesh_imgs[i, ...].cpu().numpy(), cmap = "gray")
			plt.title("Original Mesh")
			plt.colorbar()
			plt.show()
		break


def main():
	paired_paths = []
	rd_min = vd_min = 1e10
	rd_max = vd_max = -1e10

	rd_maxes, vd_maxes = [], []
	
	fun = get_img_transform("./data_stats.json", "hist_accurate_blur", 1, True)
	fun1 = get_img_transform("./data_stats.json", "hist_accurate", 1, False)
	
	"""
	fun = get_img_transform("./data_stats.json", "mesh2sfs", 1, True)
	fun1 = get_img_transform("./data_stats.json", "sfs", 1, False)
	"""
	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		rd_sum, rd_cnt, vd_sum, vd_cnt = np.zeros(40), np.zeros(40), np.zeros(40), np.zeros(40)
		errs, range_rates = [], []
		for row in tqdm(reader):
			if row["interp"] == "0" and row["human_eval"] == "1":
				
				em_idx, img_idx, try_idx = int(row["em_idx"]), int(row["img_idx"]), int(row["try_idx"])
				real_depth_path = os.path.join("depth-images", f"EM-rawdep-{em_idx}", f"{img_idx:06d}.npy")
				virtual_depth_path = os.path.join("depth-images", f"EM-newfix-{em_idx}-{try_idx}", f"{img_idx:06d}.npy")
				paired_paths.append((real_depth_path, virtual_depth_path))
				if not os.path.exists(real_depth_path):
					print(f"Real not found: {em_idx}, {img_idx}.")
					continue
				if not os.path.exists(virtual_depth_path):
					print(f"Virtual not found: {em_idx}, {try_idx}, {img_idx}.")
					continue

				rd, vd = np.load(real_depth_path), np.load(virtual_depth_path)

				rd_bars, __ = np.histogram(rd, bins = 30)
				vd_bars, __ = np.histogram(vd, bins = 30)
				rd_max, vd_max = np.argmax(rd_bars), np.argmax(vd_bars)
				rd_maxes.append(rd_max)
				vd_maxes.append(vd_max)

				#print(rd_maxes, vd_maxes)

				continue

				vd_ = transforms.GaussianBlur(21, 7)(torch.tensor(vd).unsqueeze(0)).numpy()

				rd_t, vd_t = fun1(rd).squeeze().numpy().reshape(224, 224), fun(vd).squeeze().numpy().reshape(224, 224)
				err = np.abs(rd_t - vd_t)
				range_rate = (vd.max() - vd.min()) / (rd.max() - rd.min())
				errs.append(np.mean(err))
				range_rates.append(range_rate)

				"""
				plt.scatter(rd.ravel(), vd.ravel())
				plt.xlabel("SFS Depth")
				plt.ylabel("Mesh Depth")
				plt.show()
				"""

				plt.subplot(1, 3, 1)
				plt.imshow(rd_t)
				plt.colorbar()
				plt.title("SFS Depth")
				plt.subplot(1, 3, 2)
				plt.imshow(vd_t)
				plt.title("Mesh Depth")
				plt.colorbar()
				plt.subplot(1, 3, 3)
				plt.imshow(err)
				plt.title("Absolute Error")
				plt.colorbar()
				#plt.subplot(2, 2, 4)
				#plt.scatter(rd.ravel(), vd_.ravel())
				plt.suptitle(f"MAE = {np.mean(np.abs(rd_t - vd_t)):.4f}")
				print(em_idx, img_idx)
				plt.show()
				exit()
				rd_min, rd_max = min(rd_min, rd.min()), max(rd_max, rd.max())
				vd_min, vd_max = min(vd_min, vd.min()), max(vd_max, vd.max())
				
	"""
	plt.scatter(range_rates, errs)
	print("Mean err = ", np.mean(errs))
	plt.show()
	exit()
	"""

	plt.subplot(1, 2, 1)
	plt.hist(rd_maxes)
	plt.title("RD")
	plt.subplot(1, 2, 2)
	plt.hist(vd_maxes)
	plt.title("VD")
	plt.show()
	exit()

	print("Everything found: ", len(paired_paths))
	print(f"RD: ({rd_min:.2f}, {rd_max:.2f}), VD: ({vd_min:.2f}, {vd_max:.2f}).")
	device = "cuda"
	train_dloader, test_dloader = get_simple_loaders(paired_paths)
	model = Mesh2SFS(rd_min, rd_max, vd_min, vd_max).to(device)
	#model = Mesh2SFS(rd_min, rd_max, vd_min, vd_max).to(device)
	optimiser = optim.Adam(model.parameters(), lr = 5e-3)

	#plot_one_batch(model, test_dloader, device)
	for epoch in range(20):
		#print(model.contrast_factor.item())
		for phase, dloader in [("train", train_dloader), ("test", test_dloader)]:
			if phase == "train":
				model.train()
			else:
				model.eval()
			with torch.set_grad_enabled(phase == "train"):
				sum_loss, num_loss = 0, 0
				for sfs_imgs, mesh_imgs in dloader:
					sfs_imgs, mesh_imgs = sfs_imgs.to(device).float(), mesh_imgs.to(device).float()
					loss = model(sfs_imgs, mesh_imgs)[0]
					if phase == "train":
						optimiser.zero_grad()
						loss.backward()
						optimiser.step()
					sum_loss += loss.item() * sfs_imgs.size(0)
					num_loss += sfs_imgs.size(0)
			print(f"Epoch: {epoch} {phase} loss = {sum_loss / num_loss :.4f}")
		print(model._w.item(), model._b.item())
	#plot_one_batch(model, test_dloader, device)
	
if __name__ == '__main__':
	main()
