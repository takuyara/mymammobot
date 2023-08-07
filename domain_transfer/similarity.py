import numpy as np
from utils.stats import get_num_bins

dark_threshold = 8
dark_weight = 0.2

def adjust_hist(h, eps = 1e-6):
	h = h + eps
	return h / np.sum(h)

def kl_div(p, q):
	return np.sum(p * np.log(p / q))

def kl_sim(img1, img2):
	img1, img2 = img1.flatten(), img2.flatten()
	num_bins = int((get_num_bins(img1) + get_num_bins(img2)) / 2)
	h1, __ = np.histogram(img1, bins = num_bins, density = True)
	h2, __ = np.histogram(img2, bins = num_bins, density = True)
	return -kl_div(adjust_hist(h1), adjust_hist(h2))

def corr_sim(img1, img2):
	img1, img2 = img1.flatten(), img2.flatten()
	return np.corrcoef([img1, img2])[1][0]

def mi_sim(img1, img2, num_bins = 20):
	img1, img2 = img1.flatten(), img2.flatten()
	hist2d, __, ___ = np.histogram2d(img1, img2, bins = num_bins)
	pxy = hist2d / np.sum(hist2d)
	px, py = np.sum(pxy, axis = 1), np.sum(pxy, axis = 0)
	px_py = px.reshape(-1, 1) * py.reshape(1, -1)
	indices = pxy > 0
	mi = np.sum(pxy[indices] * np.log(pxy[indices] / px_py[indices]))
	return mi

def adjusted_corr(x1, x2):
	return max(0, np.corrcoef([x1, x2])[1][0])

def comb_corr_sim(img1, img2):
	# img2 should be in virtual domain!
	img1, img2 = img1.flatten(), img2.flatten()
	if np.sum(img2 > dark_threshold) > 0:
		light_corr = adjusted_corr(img1[img2 > dark_threshold], img2[img2 > dark_threshold])
		t_dark_weight = dark_weight
	else:
		light_corr = 0
		t_dark_weight = 1
	dark_corr = adjusted_corr(img1[img2 <= dark_threshold], img2[img2 <= dark_threshold])
	return t_dark_weight * dark_corr + (1 - t_dark_weight) * light_corr
