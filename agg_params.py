import csv

def add_dict(img2data, path, interp):
	with open(path, newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			em_idx, img_idx = int(row["em_idx"]), int(row["img_idx"])
			row["interp"] = 1 if interp else 0
			if (em_idx, img_idx) not in img2data or img2data[(em_idx, img_idx)]["human_eval"] == "0":
				img2data[(em_idx, img_idx)] = row
	return reader.fieldnames + ["interp"]

def main():
	img2data = {}
	add_dict(img2data, "reg_params_interp_processed.csv", interp = True)
	headers = add_dict(img2data, "reg_params_non_interp_processed.csv", interp = False)
	print(headers)
	with open("aggred_res.csv", "w", newline = "") as f:
		writer = csv.DictWriter(f, fieldnames = headers)
		writer.writeheader()
		for em_idx, n_imgs in enumerate([2247, 2575, 2156]):
			for img_idx in range(n_imgs):
				if (em_idx, img_idx) in img2data:
					writer.writerow(img2data[(em_idx, img_idx)])
				else:
					print("Data missing: ", em_idx, img_idx)

if __name__ == '__main__':
	main()
