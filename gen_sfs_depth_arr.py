import os
import cv2
import numpy as np

base_dir = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences"
video_names = ["logfile_2015-11-13_16-04-19_Phantom_1", "logfile_2015-11-13_17-08-08_Phantom_2", "logfile_2015-11-13_17-12-48_Phantom_3"]
output_dir = "./depth-images/"
target_shape = 224

"""
min_dep, max_dep = 1e10, -1e10
for video_name in video_names:
	all_txt_path = os.path.join(base_dir, video_name, "depth_board2_enhanced")
	for txt_path in os.listdir(all_txt_path):
		depth_img = np.loadtxt(os.path.join(all_txt_path, txt_path))
		min_dep = min(np.min(depth_img), min_dep)
		max_dep = max(np.max(depth_img), max_dep)

print(min_dep, max_dep)
"""

def crop_n_reshape(img, target_shape):
	crop_size = min(img.shape[0], img.shape[1])
	x = img.shape[0] / 2 - crop_size / 2
	y = img.shape[1] / 2 - crop_size / 2
	img = img[int(x) : int(x + crop_size), int(y) : int(y + crop_size)]
	img = cv2.resize(img, (target_shape, target_shape))
	return img

for em_idx, video_name in enumerate(video_names):
	all_txt_path = os.path.join(base_dir, video_name, "depth_board2_enhanced")
	all_img_path = os.path.join(output_dir, f"EM-rawdep-{em_idx}")
	os.makedirs(all_img_path, exist_ok = True)
	for i in range(len(os.listdir(all_txt_path))):
		depth_img = np.loadtxt(os.path.join(all_txt_path, f"{i}_rect.txt")).reshape(265, 246)
		img_path = os.path.join(all_img_path, f"{i:06d}.npy")
		np.save(img_path, crop_n_reshape(depth_img, target_shape))