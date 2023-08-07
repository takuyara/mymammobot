from pystackreg import StackReg
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np

path_ct = "D:\\hope\\mymammobot\\depth-images\\CL0-fold0\\000039.png"
#path_em = "D:\\hope\\mymammobot\\depth-images\\EM-0\\000063.png"
path_em = "D:\\hope\\mymammobot\\depth-images\\CL0-fold0\\000041.png"
target_shape = 224

def load_n_reshape(path, target_shape):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	crop_size = min(img.shape[0], img.shape[1])
	x = img.shape[0] / 2 - crop_size / 2
	y = img.shape[1] / 2 - crop_size / 2
	img = img[int(x) : int(x + crop_size), int(y) : int(y + crop_size)]
	img = cv2.resize(img, (target_shape, target_shape))
	return img

ref, mov = load_n_reshape(path_ct, target_shape), load_n_reshape(path_em, target_shape)
sr = StackReg(StackReg.SCALED_ROTATION)
out = sr.register_transform(ref, mov)
plt.subplot(1, 3, 1)
plt.imshow(ref, cmap = "gray")
plt.title("CT (ref)")
plt.subplot(1, 3, 2)
plt.imshow(mov, cmap = "gray")
plt.title("EM (mov)")
plt.subplot(1, 3, 3)
plt.imshow(out, cmap = "gray")
plt.title("Out")
plt.show()
