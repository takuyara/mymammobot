import os
import cv2
import numpy as np
import pyvista as pv
from scipy import ndimage
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt

from utils.misc import randu_gen
from utils.cl_utils import load_all_cls_npy, get_unique_cl_indices, get_direction_dist_radius, index2point
from utils.geometry import arbitrary_perpendicular_vector, rotate_single_vector
from ds_gen.depth_map_generation import get_zoomed_plotter, get_depth_map, is_camera_in_bounds
from ds_gen.camera_features import static_radial_offset_gen, static_focal_radial_offset_gen, camera_params

def rotate_and_crop(img, deg, img_size):
	# Clockwise
	if not np.allclose(deg, 0):
		img = ndimage.rotate(img, -deg, reshape = False)
	st = (img.shape[0] - img_size) // 2
	return img[st : st + img_size, st : st + img_size]

def generate_rotatable_images(mesh_path, seg_cl_path, output_path, reference_path, num_samples, img_size, max_axial_len, max_radius, out_pose_only = False, norm_img = False, zoom_scale = 2 ** -0.5, suffix = "", axial_extend_rate = 0.05, radial_safe_rate = 0.96, min_depth_thres = 30, max_num_iters = 2.5):
	all_cls = load_all_cls_npy(seg_cl_path)

	plotter, zoomed_size = get_zoomed_plotter(img_size, zoom_scale)
	plotter.add_mesh(pv.read(mesh_path))

	"""
	total_volume = 0
	total_cl_len = 0
	for cl_idx, on_line_idx in unique_cl_indices:
		if on_line_idx == len(all_cls[cl_idx][0]) - 1:
			continue
		cl_orientation, axial_len, lumen_radius = get_direction_dist_radius(all_cls, (cl_idx, on_line_idx))
		total_volume += axial_len * lumen_radius ** 2
		total_cl_len += axial_len

	print(f"Volume = {total_volume:.2f}. Total len = {total_cl_len:.2f}")
	"""

	#total_volume = 18000

	out_labels = []
	out_imgs_cache = np.zeros((num_samples, zoomed_size, zoomed_size), np.float32)

	img_idx = 0
	for cl_idx, (points, radiuses) in enumerate(all_cls):
		total_axial_len = 0
		for on_line_idx in range(len(points) - 1):
			cl_point_base = points[on_line_idx]
			cl_orientation, axial_len, lumen_radius = get_direction_dist_radius(all_cls, (cl_idx, on_line_idx))
			total_axial_len += axial_len

		sum_axial_len = 0
		for on_line_idx in tqdm(range(len(points) - 1)):
			cl_point_base = points[on_line_idx]
			cl_orientation, axial_len, lumen_radius = get_direction_dist_radius(all_cls, (cl_idx, on_line_idx))
			
			if abs(axial_len) < 1e-5:
				continue

			orient_perp = arbitrary_perpendicular_vector(cl_orientation)

			axial_norm_gen = randu_gen(- axial_extend_rate * axial_len, (1 + axial_extend_rate) * axial_len)
			radial_norm_gen = static_radial_offset_gen(lumen_radius)
			focal_radial_norm_gen = static_focal_radial_offset_gen(lumen_radius)
			angle_gen = randu_gen(0, 360)

			#num_samples_on_this_cl = int(round(axial_len / total_cl_len * num_samples))
			num_samples_on_this_cl = int(num_samples / len(all_cls) / (len(points) - 1))

			"""
			if cl_idx == 0:
				num_samples_on_this_cl = int(1.5 * num_samples_on_this_cl)
			elif sum_axial_len > 100:
				num_samples_on_this_cl = 0
			"""

			#print(cl_orientation)

			num_iters = num_gen_samples = 0

			while num_iters < int(max_num_iters * num_samples_on_this_cl) and num_gen_samples < num_samples_on_this_cl:
				num_iters += 1
				axial_norm = axial_norm_gen()
				radial_norm = radial_norm_gen()
				radial_rot = angle_gen()
				focal_radial_norm = focal_radial_norm_gen()
				focal_rot = angle_gen()

				"""
				print(i, axial_norm, radial_norm, focal_radial_norm)

				axial_norm = 0
				radial_norm = 0
				focal_radial_norm = 10
				t = 56
				"""

				t_position = cl_point_base + cl_orientation * axial_norm + rotate_single_vector(orient_perp, cl_orientation, radial_rot) * radial_norm
				t_focal_point = cl_point_base + cl_orientation * (camera_params["focal_length"] + axial_norm) + rotate_single_vector(orient_perp, cl_orientation, focal_rot) * focal_radial_norm

				#t_focal_point = cl_point_base + cl_orientation * (camera_params["focal_length"] + axial_norm) + rotate_single_vector(orient_perp, cl_orientation, t) * focal_radial_norm
				#t_focal_point = cl_point_base + cl_orientation * (camera_params["focal_length"] + axial_norm)
				t_orientation = t_focal_point - t_position
				t_orientation = t_orientation / np.linalg.norm(t_orientation)


				t_up = rotate_single_vector(arbitrary_perpendicular_vector(t_orientation), t_orientation, angle_gen())

				rgb, dep = get_depth_map(plotter, t_position, t_orientation, t_up, zoom = zoom_scale, get_outputs = True)

				"""
				p.camera.position = t_position
				p.camera.focal_point = t_position + camera_params["focal_length"] * t_orientation

				plt.imshow(rgb)
				plt.show()
				exit()
				
				p.add_mesh(pv.Arrow(t_position, cl_orientation), color = "blue")
				p.add_mesh(pv.Arrow(t_position, rotate_single_vector(orient_perp, cl_orientation, t) * focal_radial_norm), color = "red")
				p.add_mesh(pv.Arrow(t_position, t_orientation), color = "green")
				p.show()

				"""
				#plt.imshow(rgb)
				#plt.show()


				if abs(np.max(np.min(rgb, axis = -1)) - 255) < 1e-2:
					continue

				if np.max(dep) < min_depth_thres:
					plt.imshow(rgb)
					plt.show()
					continue

				"""
				plt.imshow(rgb)
				plt.title("Rate: {:.2f}, Max dep {:.2f}".format(focal_radial_norm / lumen_radius, np.max(dep)))
				plt.show()
				"""

				num_gen_samples += 1
				out_pose = np.stack([t_position, t_orientation, t_up], axis = 0)
				cl_pose = np.array([cl_idx, (sum_axial_len + axial_norm) / max_axial_len, radial_norm / max_radius, radial_rot])
				all_pose = np.concatenate([cl_pose, out_pose.ravel()], axis = 0)

				out_imgs_cache[img_idx] = dep
				out_labels.append(all_pose.astype(np.float32))

				"""
				np.savetxt(os.path.join(output_path, f"{img_idx:06d}.txt"), out_pose, fmt = "%.6f")
				np.savetxt(os.path.join(output_path, f"{img_idx:06d}_clbase.txt"), cl_pose, fmt = "%.6f")
				#cv2.imwrite(os.path.join(reference_path, f"{img_idx:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
				if not out_pose_only:
					if not norm_img:
						np.save(os.path.join(output_path, f"{img_idx:06d}.npy"), dep)
						#cv2.imwrite(os.path.join(reference_path, f"{img_idx:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
					else:
						dep = ((dep - np.min(dep)) / (np.max(dep) - np.min(dep))).astype(np.float32)
						Image.fromarray(dep).save(os.path.join(output_path, f"{img_idx:06d}.tif"))
						Image.fromarray(rgb).save(os.path.join(reference_path, f"{img_idx:06d}.png"))
						#cv2.imwrite(, dep)
						#cv2.imwrite(, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
				"""
				img_idx += 1

			sum_axial_len += axial_len
			if sum_axial_len > max_axial_len:
				break
	
	out_labels = np.stack(out_labels, axis = 0)
	np.save(f"./train_{suffix}_img.npy", out_imgs_cache[ : len(out_labels), ...])
	np.save(f"./train_{suffix}_label.npy", out_labels)
	print(out_imgs_cache.shape, out_labels.shape)
