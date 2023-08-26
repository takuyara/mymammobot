import csv
import matplotlib.pyplot as plt

def main():
	id2data = {}
	lumen_radius, radial_norm, orient_norm = [], [], []
	with open("reg_params_non_interp_processed.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if int(row["human_eval"]) == 1:
				id_tup = int(row["em_idx"]), int(row["img_idx"])
				lumen_radius.append(float(row["lumen_radius"]))
				radial_norm.append(float(row["radial_norm"]))
				orient_norm.append(float(row["orient_norm"]))
				if id_tup in id2data:
					print(f"Warning: duplicated correct data for {id_tup}.")
				id2data[id_tup] = row
	plt.scatter(lumen_radius, radial_norm)
	plt.show()
	plt.scatter(lumen_radius, orient_norm)
	plt.show()

if __name__ == '__main__':
	main()
