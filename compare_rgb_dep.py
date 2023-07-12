import cv2
import numpy as np
import matplotlib.pyplot as plt

rgb_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\croped\\{}.png"
dep_path = "E:\\nn-data\\MAMMOBOT-Original\\Dataset_20151113\\Dataset_20151113\\EM_Video_sequences\\logfile_2015-11-13_16-04-19_Phantom_1\\depth_board2_enhanced\\{}_rect.txt"
img_id = 100

rgb_path, dep_path = rgb_path.format(img_id), dep_path.format(img_id)
rgb_img = cv2.imread(rgb_path)
print(rgb_img.shape)
dep_path = np.loadtxt(dep_path).reshape(rgb_img.shape[ : 2])

f, axarr = plt.subplots(1, 2)
axarr[0].imshow(rgb_img)
axarr[1].imshow(dep_path, cmap = "coolwarm")
plt.show()
