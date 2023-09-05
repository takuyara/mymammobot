import os
import numpy as np
def get_data(paths, prefix):
	img_list, label_list = [], []
	for this_path in paths:
		for npy_path in os.listdir(this_path):
			if npy_path.endswith(".npy"):
				npy_path = os.path.join(this_path, npy_path)
				img = np.load(npy_path)
				label = int(np.loadtxt(npy_path.replace(".npy", "_clbase.txt")).ravel()[0])
				img_list.append(img)
				label_list.append(label)
	img_list, label_list = np.stack(img_list, axis = 0), np.stack(label_list, axis = 0)
	np.save(f"{prefix}_img.npy", img_list)
	np.save(f"{prefix}_label.npy", label_list)

get_data(["virtual_dataset/full/train"], "train")
get_data(["real_dataset/confirmed/real-0", "real_dataset/confirmed/real-1", "real_dataset/confirmed/real-2"], "val")
