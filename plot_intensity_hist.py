import numpy as np
import matplotlib.pyplot as plt

M = 100000
n_bins = 20
eps = 1e-6

def compute_mean_std(x, c):
	mean = np.sum(x * c)
	print(x, mean)
	print((x - mean) ** 2)
	std = np.sum(((x - mean) ** 2) * c) ** 0.5
	return mean, std

def get_values(x, c):
	res = []
	for tx, tc in zip(x, c):
		res.extend([tx for __ in range(int(tc * M))])
	return np.array(res)


def main():
	em_hist = np.load("em_hist.npy")
	ct_hist = np.load("ct_hist.npy")

	plt.subplot(1, 2, 1)
	plt.bar(range(256), em_hist)
	plt.ylim(0, 0.04)
	plt.title("EM Intensity Histogram")
	plt.subplot(1, 2, 2)
	plt.bar(range(256), ct_hist)
	plt.ylim(0, 0.04)
	plt.title("CT Intensity Histogram")
	plt.show()


	x_ct = np.arange(256)
	em_v = get_values(x_ct, em_hist)
	ct_v = get_values(x_ct, ct_hist)
	bin_lims = (np.arange(n_bins) + 1) * 256 / n_bins
	ct_hist = get_hist(ct_v, bin_lims)
	best_kld = kl_div(get_hist(em_v, bin_lims), ct_hist)
	for w in np.arange(0.5, 2, 0.01):
		for b in np.arange(-20, 20, 0.1):
			em_v_1 = em_v * w + b
			em_hist_new = get_hist(em_v_1, bin_lims)
			this_kld = kl_div(em_hist_new, ct_hist)
			if this_kld < best_kld:
				print("New kld: ", this_kld, w, b)
				best_kld = this_kld
				best_w, best_b = w, b
	print(best_w, best_b, best_kld)


	"""
	m_em, s_em = compute_mean_std(x_ct, em_hist)
	m_ct, s_ct = compute_mean_std(x_ct, ct_hist)
	print("EM: {:.2f}+-{:.2f}".format(m_em, s_em))
	print("CT: {:.2f}+-{:.2f}".format(m_ct, s_ct))
	x_em = (x_ct - m_em) / s_em * s_ct + m_ct
	h_ct = get_values(x_ct, ct_hist)
	h_em = get_values(x_em, em_hist)
	print("Adj EM: {:.2f}+-{:.2f}".format(np.mean(h_em), np.mean(h_ct)))
	plt.subplot(1, 2, 1)
	plt.hist(h_em, bins = 256)
	plt.xlim(0, 256)
	plt.ylim(0, 0.1 * M)
	plt.title("Adjusted EM Histogram")
	plt.subplot(1, 2, 2)
	plt.hist(h_ct, bins = 256)
	plt.xlim(0, 256)
	plt.ylim(0, 0.1 * M)
	plt.title("CT Histogram")
	plt.show()
	"""

	"""
	plt.subplot(1, 2, 1)
	plt.bar(range(256), em_hist)
	plt.ylim(0, 0.04)
	plt.title("EM Intensity Histogram")
	plt.subplot(1, 2, 2)
	plt.bar(range(256), ct_hist)
	plt.ylim(0, 0.04)
	plt.title("CT Intensity Histogram")
	plt.show()
	"""

if __name__ == '__main__':
	main()
