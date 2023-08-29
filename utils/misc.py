import numpy as np

def str_to_arr(s):
	s = s[1 : -1].split()
	#assert len(s) == 3
	return np.array([float(x) for x in s])

def randu_gen(a, b):
	def randu():
		return np.random.rand() * (b - a) + a
	return randu
