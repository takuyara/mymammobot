import os
import csv

em_indices = [0, 1, 2]
try_indices = [0, 1, 2]
step_size = 2

all_required = []

for em_idx in em_indices:
	em_path = os.path.join("./depth-images", f"EM-{em_idx}")
	for img_idx in range(0, len(os.listdir(em_path)) // 2, step_size):
		img_found = False
		for try_idx in try_indices:
			if os.path.exists(os.path.join("./depth-images", f"EM-newfix-{em_idx}-{try_idx}", f"{img_idx:06d}.npy")):
				img_found = True
				break
		if not img_found:
			all_required.append([em_idx, img_idx])

with open("ill-registered.csv", "w", newline = "") as f:
	writer = csv.writer(f)
	writer.writerows(all_required)
