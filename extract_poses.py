import os
import shutil

path = "./virtual_dataset/single_image"
new_path = "./virtual_dataset/single_image_pose_only"

for partition in ["train", "val"]:
	os.makedirs(os.path.join(new_path, partition), exist_ok = True)
	for file in os.listdir(os.path.join(path, partition)):
		if file.endswith(".txt"):
			shutil.copy(os.path.join(path, partition, file), os.path.join(new_path, partition, file))