import os
import cv2
import numpy as np

base_dir = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences"
video_names = ["logfile_2015-11-13_16-04-19_Phantom_1", "logfile_2015-11-13_17-08-08_Phantom_2", "logfile_2015-11-13_17-12-48_Phantom_3"]

min_dep, max_dep = 1e10, -1e10
for video_name in video_names:
	all_txt_path = os.path.join(base_dir, video_name, "depth_board2_enhanced")
	for txt_path in os.listdir(all_txt_path):
		depth_img = np.loadtxt(os.path.join(all_txt_path, txt_path))
		min_dep = min(np.min(depth_img), min_dep)
		max_dep = max(np.max(depth_img), max_dep)

print(min_dep, max_dep)

for video_name in video_names:
	all_txt_path = os.path.join(base_dir, video_name, "depth_board2_enhanced")
	all_img_path = os.path.join(base_dir, video_name, "depth_img")
	os.makedirs(all_img_path, exist_ok = True)
	for txt_path in os.listdir(all_txt_path):
		depth_img = np.loadtxt(os.path.join(all_txt_path, txt_path)).reshape(265, 246)
		img_path = os.path.join(all_img_path, txt_path.replace("_rect.txt", ".png"))
		depth_img = (depth_img - min_dep) / (max_dep - min_dep) * 255
		depth_img = depth_img.astype(np.uint8)
		cv2.imwrite(img_path, depth_img)
