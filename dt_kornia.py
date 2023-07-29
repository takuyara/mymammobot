import matplotlib.pyplot as plt
import cv2
import numpy as np
import kornia as K
import kornia.geometry as KG
import torch
from torch import nn

path_ct = "D:\\hope\\mymammobot\\depth-images\\CL0-fold0\\000039.png"
path_em = "D:\\hope\\mymammobot\\depth-images\\EM-0\\000063.png"
#path_em = "D:\\hope\\mymammobot\\depth-images\\CL0-fold0\\000041.png"
target_shape = 224
pyramid_levels = 5

def load_image(path, target_shape):
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	tensor = K.image_to_tensor(img, None).float() / 255.0
	tensor = K.color.bgr_to_rgb(tensor)
	tensor = KG.transform.resize(tensor, target_shape)
	tensor = KG.transform.center_crop(tensor, (target_shape, target_shape))
	return tensor

def write_image(tensor):
	return K.tensor_to_image((tensor * 255.0).byte())

losses = []
def corr_loss(t1, t2, reduction):
	dt = torch.stack([t1.flatten(), t2.flatten()], axis = 0)
	corr = -torch.corrcoef(dt)[1][0]
	losses.append(corr.item())
	return torch.ones_like(t1) * corr

ct_img, em_img = load_image(path_ct, target_shape), load_image(path_em, target_shape)
registrator = KG.ImageRegistrator("similarity", loss_fn = corr_loss, num_iterations = 1000, pyramid_levels = pyramid_levels)
#registrator = KG.ImageRegistrator("similarity")


homo = registrator.register(em_img, ct_img)

out = registrator.warp_src_into_dst(em_img)

#print(nn.L1Loss()(mov, out), nn.L1Loss()(ref, out1), nn.L1Loss()(ref, mov))

#print(losses, len(losses))

#plt.plot(range(len(losses)), losses)
losses = losses[ : len(losses) // pyramid_levels * pyramid_levels]
losses = np.array(losses).reshape(-1, pyramid_levels)
losses = np.mean(losses, axis = -1)
plt.plot(losses)
plt.xlabel("# Iteration")
plt.ylabel("Negative Correlation")
plt.title("Negative Correlation Loss Curve for Sample Registration")
plt.show()


plt.subplot(1, 3, 1)
plt.imshow(write_image(ct_img), cmap = "gray")
plt.title("CT")
plt.subplot(1, 3, 2)
plt.imshow(write_image(em_img), cmap = "gray")
plt.title("EM")
plt.subplot(1, 3, 3)
plt.imshow(write_image(out), cmap = "gray")
plt.title("EM (transformed to CT)")
plt.show()


