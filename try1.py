import numpy as np
import matplotlib.pyplot as plt
imgs = np.load("train_img.npy")
labels = np.load("train_label.npy")
while True:
	idx = int(input("INPUT"))
	plt.imshow(imgs[idx, ...])
	plt.title("{} {:.4f}".format(labels[idx, 0], labels[idx, 1]))
	plt.show()
