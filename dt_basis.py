import os
import csv
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from domain_transfer.transformation import get_simple_loaders, SFS2Mesh, Mesh2SFS

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
	with open("aggred_res.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
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
				rd_min, rd_max = min(rd_min, rd.min()), max(rd_max, rd.max())
				vd_min, vd_max = min(vd_min, vd.min()), max(vd_max, vd.max())
	print("Everything found: ", len(paired_paths))
	print(f"RD: ({rd_min:.2f}, {rd_max:.2f}), VD: ({vd_min:.2f}, {vd_max:.2f}).")
	device = "cuda"
	train_dloader, test_dloader = get_simple_loaders(paired_paths)
	model = Mesh2SFS(rd_min, rd_max, vd_min, vd_max).to(device)
	#model = Mesh2SFS(rd_min, rd_max, vd_min, vd_max).to(device)
	optimiser = optim.Adam(model.parameters(), lr = 1e-3)
	#plot_one_batch(model, test_dloader, device)
	for epoch in range(20):
		#print(model.contrast_factor.item())
		for phase, dloader in [("train", train_dloader), ("test", test_dloader)]:
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
	#plot_one_batch(model, test_dloader, device)
	print(model._w.item(), model._b.item())
if __name__ == '__main__':
	main()
