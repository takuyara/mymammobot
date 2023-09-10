import os
import numpy as np

#max_axial_len = 120
max_axial_len = 106
max_radius = 10

def get_data(paths, prefix):
	img_list, label_list = [], []
	for this_path in paths:
		for npy_path in os.listdir(this_path):
			if npy_path.endswith(".npy"):
				npy_path = os.path.join(this_path, npy_path)
				img = np.load(npy_path)
				label_cl = np.loadtxt(npy_path.replace(".npy", "_clbase.txt")).ravel()
				label_pose = np.loadtxt(npy_path.replace(".npy", ".txt")).ravel()
				img_list.append(img)
				label_list.append(np.concatenate([label_cl, label_pose], axis = 0))
	img_list, label_list = np.stack(img_list, axis = 0), np.stack(label_list, axis = 0)
	label_list = np.stack(label_list, axis = 0)
	label_list[ : , 1] = label_list[ : , 1] / max_axial_len
	label_list[ : , 2] = label_list[ : , 2] / max_radius
	np.save(f"{prefix}_img.npy", img_list)
	np.save(f"{prefix}_label.npy", label_list)

#get_data(["virtual_dataset/full/train"], "train")
#get_data(["real_dataset/confirmed/real-0", "real_dataset/confirmed/real-1", "real_dataset/confirmed/real-2"], "val_nag")
#get_data(["real_dataset/all/real-0", "real_dataset/all/real-1", "real_dataset/all/real-2"], "val_all")
get_data(["real_dataset/all/real-0"], "val_all_0")
get_data(["real_dataset/all/real-1"], "val_all_1")
get_data(["real_dataset/all/real-2"], "val_all_2")
