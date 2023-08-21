import numpy as np
import scipy.stats as st
import warnings
from scipy.stats._continuous_distns import _distn_names

def get_num_bins(x):
	# Freedman-Diaconis Rule
	h = 2 * (np.quantile(x, 0.75) - np.quantile(x, 0.25)) * (len(x) ** (-1 / 3))
	return int((np.max(x) - np.min(x)) / h)

def fit_best_distrib(x):
	densities, bin_edges = np.histogram(x, bins = get_num_bins(x), density = True)
	bin_centres = (bin_edges + np.roll(bin_edges, -1))[ : -1] / 2
	all_distribs = []
	for distrib_name in _distn_names:
		if distrib_name in ["levy_stable", "studentized_range"]:
			continue
		distrib = getattr(st, distrib_name)
		try:
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore")
				params = distrib.fit(x)
				loc, scale, others = params[-2], params[-1], params[ : -2]
				bin_centre_pdf = distrib.pdf(bin_centres, loc = loc, scale = scale, *others)
				sse = np.linalg.norm(bin_centre_pdf - densities)
				all_distribs.append((sse, distrib_name, distrib, params))
		except Exception as e:
			pass
	return min(all_distribs)

def get_pdf_curve(distrib, params, num_points = 10000, eps = 1e-4):
	loc, scale, others = params[-2], params[-1], params[ : -2]
	st, ed = tuple([distrib.ppf(vl, *others, loc = loc, scale = scale) for vl in [eps, 1 - eps]])
	x = np.linspace(st, ed, num_points)
	y = distrib.pdf(x, loc = loc, scale = scale, *others)
	return x, y

def weighted_mean(x, w):
	return (x * w).sum() / w.sum()

def weighted_cov(x, y, w):
	return (w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))).sum() / w.sum()

def weighted_corr(x, y, w):
	top = weighted_cov(x, y, w)
	bot = (weighted_cov(x, x, w) * weighted_cov(y, y, w)) ** 0.5
	top, bot = float(top), float(bot)
	if abs(bot) < 1e-10:
		return -2
	return top / bot
