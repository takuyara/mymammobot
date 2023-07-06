def get_dir_list(path):
	res = []
	with open(path) as f:
		for row in f.readlines():
			res.append(row.strip())
	return res
