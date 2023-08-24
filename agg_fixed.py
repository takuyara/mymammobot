import os
import csv
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.pose_utils import get_output_array

from vtkmodules.vtkRenderingCore import vtkCamera

def str_to_arr(s):
	s = s[1 : -1].split()
	assert len(s) == 3
	return np.array([float(x) for x in s])

def main():
	trys_data = {}
	with open("reg_params_no_interp_processed.csv", newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			em_idx, img_idx, try_idx = int(row["em_idx"]), int(row["img_idx"]), int(row["try_idx"])
			position = str_to_arr(row["position"])
			orientation = str_to_arr(row["orientation"])
			up = str_to_arr(row[7])
			#print(np.rad2deg(np.arccos(np.dot(orientation, up) / np.linalg.norm(orientation) / np.linalg.norm(up))))
			#print(np.linalg.norm(orientation))
			#orientation = orientation / np.linalg.norm(orientation)
			quaternion = compute_rotation_quaternion(camera_params["forward_direction"], orientation)

			focal_point = position + camera_params["focal_length"] * orientation


			camera = vtkCamera()
			camera.SetPosition(*position)
			camera.SetFocalPoint(*focal_point)
			camera.SetViewUp(*up)
			camera.OrthogonalizeViewUp()
			up1 = camera.GetViewUp()
			#print(up, up1)

			print(np.rad2deg(np.arccos(np.dot(orientation, up1) / np.linalg.norm(orientation) / np.linalg.norm(up1))))

			"""

			rotated = R.from_quat(quaternion).apply(camera_params["forward_direction"])
			rotated = rotated / np.linalg.norm(rotated)
			
			assert np.allclose(rotated, orientation)
			"""

			trys_data[(em_idx, img_idx, try_idx)] = get_output_array(position, orientation)

	exit()
	
	with open("select_result_contour.csv", newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			em_idx, try_idx, img_idx, label = int(row[0]), int(row[1]), int(row[2]), int(row[3])
			if label == 1:
				out_path = os.path.join("./depth-images", f"EM-fix-{em_idx}", f"{img_idx:06d}")
				orig_path = os.path.join("./depth-images", f"EM-newfix-{em_idx}-{try_idx}", f"{img_idx:06d}")
				shutil.copy(orig_path + ".png", out_path + ".png")
				shutil.copy(orig_path + ".npy", out_path + ".npy")
				np.savetxt(out_path + ".txt", trys_data[(em_idx, img_idx, try_idx)], fmt = "%.6f")

if __name__ == '__main__':
	main()
