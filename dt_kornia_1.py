import matplotlib.pyplot as plt
import cv2
import numpy as np
import kornia as K
import kornia.geometry as KG
import torch
from torch import nn
from image_registration import ImageRegistrator

path_ct = "D:\\hope\\mymammobot\\depth-images\\CL0-fold0\\000039.png"
path_em = "D:\\hope\\mymammobot\\depth-images\\EM-0\\000063.png"
#path_em = "D:\\hope\\mymammobot\\depth-images\\CL0-fold0\\000041.png"
target_shape = 224
pyramid_levels = 5
batch_size = 16
device = "cuda"

def load_image(path, target_shape):
	img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	tensor = K.image_to_tensor(img, None).float() / 255.0
	tensor = KG.transform.resize(tensor, target_shape)
	tensor = KG.transform.center_crop(tensor, (target_shape, target_shape))
	return tensor

def write_image(tensor):
	return K.tensor_to_image((tensor * 255.0).byte())


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


plt.subplot(1, 3, 1)
plt.imshow(write_image(ct_img[0]), cmap = "gray")
plt.title("CT")
plt.subplot(1, 3, 2)
plt.imshow(write_image(em_img[0]), cmap = "gray")
plt.title("EM")
plt.subplot(1, 3, 3)
plt.imshow(write_image(out[0]), cmap = "gray")
plt.title("EM (transformed to CT)")
plt.show()


