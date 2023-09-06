import numpy as np
import matplotlib.pyplot as plt
imgs = np.load("train_imgs_new.npy")
labels = np.load("train_labels_new.npy")
while True:
	idx = int(input("INPUT"))
	plt.imshow(imgs[idx, ...])
	plt.title("{} {:.4f}".format(labels[idx, 0], labels[idx, 1]))
	plt.show()
