import csv
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from utils.misc import str_to_arr
from utils.geometry import rotate_single_vector, arbitrary_perpendicular_vector
from utils.cl_utils import index2point, get_direction_dist_radius, load_all_cls
from ds_gen.depth_map_generation import get_depth_map

def main():
	all_cls = load_all_cls("./CL")

	p = pv.Plotter(off_screen = True, window_size = (224, 224))
	p.add_mesh(pv.read("./meshes/Airway_Phantom_AdjustSmooth.stl"))

	x, y = [], []

	id2data = {}
	with open("reg_params_non_interp_processed.csv", newline = "") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if int(row["human_eval"]) == 1:
				id_tup = int(row["em_idx"]), int(row["img_idx"])
				if id_tup in id2data:
					print(f"Warning: duplicated correct data for {id_tup}.")
				id2data[id_tup] = row
				x.append(float(row["lumen_radius"]))
				y.append(float(row["focal_radial_norm"]))
				continue

				cl_indices = int(row["cl_idx"]), int(row["on_line_idx"])
				cl_point = index2point(all_cls, cl_indices)
				cl_orientation, axial_len, lumen_radius = get_direction_dist_radius(all_cls, cl_indices)

				pt_confirm = cl_point + rotate_single_vector(arbitrary_perpendicular_vector(cl_orientation), cl_orientation, float(row["radial_rot"])) * float(row["radial_norm"]) + cl_orientation * float(row["axial_norm"])
				or_confirm = cl_orientation + rotate_single_vector(arbitrary_perpendicular_vector(cl_orientation), cl_orientation, float(row["orient_rot"])) * float(row["orient_norm"])
				or_confirm = or_confirm / np.linalg.norm(or_confirm)

				if not np.allclose(pt_confirm, str_to_arr(row["position"])):
					print("PE")
				if not np.allclose(or_confirm, str_to_arr(row["orientation"])):
					print("OE")

				if float(row["orient_norm"]) > 0.5:
					img_this = get_depth_map(p, pt_confirm, or_confirm, arbitrary_perpendicular_vector(or_confirm), get_outputs = True)[0]
					img_orig_orient = get_depth_map(p, pt_confirm, cl_orientation, arbitrary_perpendicular_vector(cl_orientation), get_outputs = True)[0]
					img_orig = get_depth_map(p, cl_point, cl_orientation, arbitrary_perpendicular_vector(cl_orientation), get_outputs = True)[0]
					plt.subplot(1, 3, 1)
					plt.imshow(img_this)
					plt.title("This image")
					plt.subplot(1, 3, 2)
					plt.imshow(img_orig_orient)
					plt.title("This pos, orig orient")
					plt.subplot(1, 3, 3)
					plt.imshow(img_orig)
					plt.title("Orig pos, orient")
					plt.suptitle("Radial {:.0f} deg * {:.4f}, Orient {:.0f} deg * {:.4f}".format(float(row["radial_rot"]), float(row["radial_norm"]), float(row["orient_rot"]), float(row["orient_norm"])))
					plt.show()

	plt.scatter(x, y)
	plt.show()
if __name__ == '__main__':
	main()
