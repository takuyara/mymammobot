def get_dir_list(path):
	if path.endswith(".txt"):
		res = []
		with open(path) as f:
			for row in f.readlines():
				res.append(row.strip())
		return res
	else:
		return [path]