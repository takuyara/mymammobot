import os
import cv2
import numpy as np
from tqdm import tqdm

base_path = "D:\\hope\\mymammobot\\depth-images"
em_all_path = "D:\\hope\\mymammobot\\depth-images\\EM-0"
target_shape = 224
candidate_distance = 5
maximum_candidates = 1
pyramid_levels = 3
lr = 1e-4
tolerance = 1e-6
num_iters = 2000
batch_size = 4
device = "cuda"
transform_type = "rigid"

def main():
	for this_em in os.listdir(base_path):
		if this_em.startswith("EM"):
			this_em = os.path.join(base_path, this_em)
			os.makedirs(this_em + "-AG", exist_ok = True)
			for i in tqdm(range(len(os.listdir(this_em)) // 2)):
				em_path = os.path.join(this_em, f"{i:06d}.png")
				if not os.path.exists(em_path):
					break
				img = cv2.imread(em_path, cv2.IMREAD_GRAYSCALE)
				img = img * 1.03 + 10.20
				cv2.imwrite(os.path.join(this_em + "-AG", f"{i:06d}.png"), img)

if __name__ == '__main__':
	main()



