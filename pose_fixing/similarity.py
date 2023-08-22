import cv2
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from scipy.spatial.distance import directed_hausdorff as DHD

from utils.stats import get_num_bins, weighted_corr

dark_threshold_r = 3
dark_threshold_v = 8
dark_weight = 0.7
light_weight = 1.5

def adjust_hist(h, eps = 1e-6):
	h = h + eps
	return h / np.sum(h)

def kl_div(p, q):
	return np.sum(p * np.log(p / q))

def kl_sim(img1, img2):
	img1, img2 = img1.ravel(), img2.ravel()
	num_bins = int((get_num_bins(img1) + get_num_bins(img2)) / 2)
	h1, __ = np.histogram(img1, bins = num_bins, density = True)
	h2, __ = np.histogram(img2, bins = num_bins, density = True)
	return -kl_div(adjust_hist(h1), adjust_hist(h2))

def corr_sim(img1, img2):
	img1, img2 = img1.ravel(), img2.ravel()
	return np.corrcoef([img1, img2])[1][0]

def mi_sim(img1, img2, num_bins = 20):
	img1, img2 = img1.ravel(), img2.ravel()
	hist2d, __, ___ = np.histogram2d(img1, img2, bins = num_bins)
	pxy = hist2d / np.sum(hist2d)
	px, py = np.sum(pxy, axis = 1), np.sum(pxy, axis = 0)
	px_py = px.reshape(-1, 1) * py.reshape(1, -1)
	indices = pxy > 0
	mi = np.sum(pxy[indices] * np.log(pxy[indices] / px_py[indices]))
	return mi

def get_dark_threshold(x, bins = 30, rate = 0.25):
	h, bin_edges = np.histogram(x.ravel(), bins = bins, density = True)
	first_outlier_idx = len(h) - np.argmax(np.flip((h > np.max(h) * rate).astype(int)))
	return bin_edges[first_outlier_idx]

def comb_corr_sim(img1, img2, light_mask_1 = None, light_mask_2 = None, light_weight = light_weight, dark_weight = dark_weight):
	# img1: SFS, img2: Mesh
	img1, img2 = img1.ravel(), img2.ravel()
	if light_mask_1 is None:
		light_mask_1 = get_light_mask(img1)
	else:
		light_mask_1 = light_mask_1.ravel()
	if light_mask_2 is None:
		light_mask_2 = get_light_mask(img2)
	else:
		light_mask_2 = light_mask_2.ravel()

	weights = np.ones_like(img1) * dark_weight
	weights[np.logical_or(light_mask_1, light_mask_2)] = light_weight
	return weighted_corr(img1, img2, weights)

def unweighted_corr_sim(img1, img2):
	img1, img2 = img1.ravel(), img2.ravel()
	return weighted_corr(img1, img2, np.ones_like(img1))

def reg_mse_sim(img1, img2, light_weight = light_weight, dark_weight = dark_weight, ret_value = "r2"):
	img1, img2 = img1.flatten(), img2.flatten()
	dark_thres_1 = get_dark_threshold(img1)
	dark_thres_2 = get_dark_threshold(img2)
	weights = np.ones_like(img1) * light_weight
	weights[np.logical_and(img1 < dark_thres_1, img2 < dark_thres_2)] = dark_weight
	img1, img2 = img1.reshape(-1, 1), img2.reshape(-1, 1)
	#rd1, vd1 = rd1.reshape(-1, 1), vd1.reshape(-1, 1)
	reg = LR().fit(img1, img2, weights)
	if ret_value == "r2":
		return reg.score(img1, img2, weights)
	elif ret_value == "weighted_mse":
		return -np.sum(((img2 - reg.predict(img1)) ** 2).flatten() * weights) / np.sum(weights)
	else:
		raise NotImplementedError
	#err = np.sum(((img2 - reg.predict(img1)) ** 2).flatten() * weights) / np.sum(weights)
	#return -err
	#return reg.score(img1, img2, weights)

def get_light_mask(img, rate = 0.25, quantile = None):
	if quantile is None:
		thres = get_dark_threshold(img, rate = rate)
	else:
		thres = np.quantile(img.ravel(), quantile)
	return (img > thres).astype(np.uint8)

def get_contours(img):
	contours, __ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if len(contours) == 0:
		return None
	contours = np.concatenate(contours, axis = 0).reshape(-1, 2)
	return contours

def cache_base_data(img, rate = 0.25):
	light_mask = get_light_mask(img, rate = rate)
	quantile = 1 - np.sum(light_mask) / len(img.ravel())
	contours = get_contours(light_mask)
	return contours, quantile, light_mask

def cache_multiscale_base_data(img, scale_rates):
	base_contours_list, quantile_list, base_light_mask_list = [], [], []
	for rate in scale_rates:
		t_c, t_q, t_lm = cache_base_data(img, rate = rate)
		base_contours_list.append(t_c)
		quantile_list.append(t_q)
		base_light_mask_list.append(t_lm)
	return base_contours_list, quantile_list, base_light_mask_list

def contour_sim(img, base_contours, quantile, light_mask = None):
	if light_mask is None:
		light_mask = get_light_mask(img, quantile = quantile)
	contours = get_contours(light_mask)

	if contours is None:
		return None
	hausdorff_dist = max(DHD(contours, base_contours)[0], DHD(base_contours, contours)[0])
	return -hausdorff_dist

def draw_contours(img, quantile):
	light_mask = get_light_mask(img, quantile = quantile)
	contours, __ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	img_normed = (img - img.min()) / (img.max() - img.min())
	img_normed = img_normed.reshape(*img_normed.shape, 1).astype(np.float32)
	#print(img_normed, img_normed.shape)
	rgb_img = cv2.cvtColor(img_normed, cv2.COLOR_GRAY2BGR)
	contours_img = cv2.drawContours(rgb_img, contours, -1, (0, 255, 0), 3)
	return contours_img

def contour_corr_sim(img, base_img, base_contours, quantile, base_light_mask):
	light_mask = get_light_mask(img, quantile = quantile)
	cont_sim = contour_sim(img, base_contours, quantile, light_mask)
	corr_sim = comb_corr_sim(base_img, img, base_light_mask, light_mask)
	if cont_sim is None or corr_sim is None:
		return None
	return cont_sim, corr_sim

def multiscale_contour_corr_sim(img, base_img, base_contours_list, quantile_list, base_light_mask_list):
	sum_cont_sim, sum_corr_sim = 0, 0
	for base_contours, quantile, base_light_mask in zip(base_contours_list, quantile_list, base_light_mask_list):
		res = contour_corr_sim(img, base_img, base_contours, quantile, base_light_mask)
		if res is None:
			return None
		sum_cont_sim += res[0]
		sum_corr_sim += res[1]
	return sum_cont_sim / len(quantile_list), sum_corr_sim / len(quantile_list)
