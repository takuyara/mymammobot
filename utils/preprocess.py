import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from scipy import ndimage

from ds_gen.rotatable_single_images import rotate_and_crop
from ds_gen.depth_map_generation import get_depth_map
from utils.misc import randu_gen
from utils.pose_utils import compute_rotation_quaternion, get_3dof_quat, revert_quat, camera_pose_to_train_pose, get_6dof_pose_label
from utils.geometry import rotate_single_vector, arbitrary_perpendicular_vector

def load_img_pose(base_path, idx, plotter):
	pose = np.loadtxt(os.path.join(base_path, f"{idx:06d}.txt"))
	pose_tup = pose[0], pose[1], pose[2]
	if plotter is not None:
		img = get_depth_map(plotter, *pose_tup)
	else:
		img = np.load(os.path.join(base_path, f"{idx:06d}.npy"))
	if np.all(np.isnan(img)):
		img = np.random.randn(img.shape) * 0.01
	else:
		img = np.nan_to_num(img, nan = img.max())
	pose = camera_pose_to_train_pose(*pose_tup)
	pose = get_6dof_pose_label(pose)
	return img, torch.tensor(pose)

def random_rotate_camera(img, pose, img_size, plotter = None, deg = 0, zoom_scale = 1.0):
	position, orientation = pose[0, ...], pose[1, ...]
	if deg != 0 or len(pose) < 3:
		up = rotate_single_vector(arbitrary_perpendicular_vector(orientation), orientation, deg)
	else:
		up = pose[2, ...]
	if plotter is None:
		#img = rotate_and_crop(img, deg, img_size)
		assert deg == 0
	else:
		img = get_depth_map(plotter, position, orientation, up, zoom = zoom_scale)
	if np.all(np.isnan(img)):
		img = np.random.randn(img.shape) * 0.01
	else:
		img = np.nan_to_num(img, nan = img.max())
	pose = camera_pose_to_train_pose(position, orientation, up)
	return img, pose

def cv2_clipped_zoom(img, zoom_factor=0):

	"""
	Center zoom in/out of the given image and returning an enlarged/shrinked view of 
	the image without changing dimensions
	------
	Args:
		img : ndarray
			Image array
		zoom_factor : float
			amount of zoom as a ratio [0 to Inf). Default 0.
	------
	Returns:
		result: ndarray
		   numpy ndarray of the same shape of the input img zoomed by the specified factor.		  
	"""
	if zoom_factor == 0:
		return img


	height, width = img.shape[:2] # It's also the final desired shape
	new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
	
	### Crop only the part that will remain in the result (more efficient)
	# Centered bbox of the final desired size in resized (larger/smaller) image coordinates
	y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
	y2, x2 = y1 + height, x1 + width
	bbox = np.array([y1,x1,y2,x2])
	# Map back to original image coordinates
	bbox = (bbox / zoom_factor).astype(np.int)
	y1, x1, y2, x2 = bbox
	cropped_img = img[y1:y2, x1:x2]
	
	# Handle padding when downscaling
	resize_height, resize_width = min(new_height, height), min(new_width, width)
	pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
	pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
	pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
	
	result = cv2.resize(cropped_img, (resize_width, resize_height))
	result = np.pad(result, pad_spec, mode='constant')
	assert result.shape[0] == height and result.shape[1] == width
	return result


def get_img_transform(data_stats_path, method, n_channels, train, args):
	stats = json.load(open(data_stats_path))
	if method in ["sfs2mesh", "mesh2sfs", "sfs", "mesh"]:
		if method in ["sfs2mesh", "mesh"]:
			img_mean, img_std = stats["mesh_mean"], stats["mesh_std"]
		else:
			img_mean, img_std = stats["sfs_mean"], stats["sfs_std"]
		if method == "sfs2mesh":
			_w, _b = stats["sfs2mesh_weight"], stats["sfs2mesh_bias"]
		elif method == "mesh2sfs":
			#_w, _b = stats["mesh2sfs_weight"], stats["mesh2sfs_bias"]
			_w, _b = 0.13236457109451294, 2.5300467014312744
		else:
			_w, _b = 1, 0
		def reshape_n_norm(img):
			img = torch.tensor(img).float().unsqueeze(0)
			if method == "mesh2sfs":
				img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img)
			img = img * _w + _b
			#img = (img - img_mean) / img_std
			return img.repeat(n_channels, 1, 1)
		return reshape_n_norm
	elif method == "quantile":
		def img_to_quantile(img):
			orig_shape = img.shape
			img = img.ravel()
			q = np.argsort(img)
			q = np.argsort(q)
			q = q / len(q)
			q = q.reshape(orig_shape)
			if train:
				#sigma_blur = randu_gen(2.0, 2.3)()
				sigma_blur = 2.0
				sigma_intensity = 0.2
				q = q + np.random.randn(*q.shape) * sigma_intensity
				q = ndimage.gaussian_filter(q, sigma = sigma_blur)
			q = torch.tensor(q).float().unsqueeze(0).repeat(n_channels, 1, 1)
			return q
		return img_to_quantile
	elif method == "quantile_sfs":
		ts = transforms.Compose([transforms.CenterCrop(190), transforms.Resize(224)])
		def img_to_quantile_sfs(img):
			orig_shape = img.shape
			img = img.ravel()
			q = np.argsort(img)
			q = np.argsort(q)
			q = q / len(q)
			q = q.reshape(orig_shape)
			if train:
				#sigma_blur = randu_gen(2.0, 2.3)()
				sigma_blur = 2.0
				sigma_intensity = 0.1
				q = q + np.random.randn(*q.shape) * sigma_intensity
				q = ndimage.gaussian_filter(q, sigma = sigma_blur)
			q = torch.tensor(q).float().unsqueeze(0).repeat(n_channels, 1, 1)
			q = ts(q)
			return q
		return img_to_quantile_sfs
	elif method == "hist_simple":
		def img_to_hist_simple(img, bins = 30):
			img = torch.tensor(img).float().unsqueeze(0)
			img = (img - img.min()) / (img.max() - img.min())
			img = torch.floor(img * bins) / bins
			return img.repeat(n_channels, 1, 1)
		return img_to_hist_simple
	elif method == "hist_simple_blur":
		def img_to_hist_simple_blur(img, bins = 30):
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img)
			img = (img - img.min()) / (img.max() - img.min())
			img = torch.floor(img * bins) / bins
			return img.repeat(n_channels, 1, 1)
		return img_to_hist_simple_blur
	elif method == "hist_accurate":
		def img_to_hist_accurate(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = (img - img.min()) / (img.max() - img.min())
			return img.repeat(n_channels, 1, 1)
		return img_to_hist_accurate
	elif method == "hist_accurate_blur":
		def img_to_hist_accurate_blur(img):
			if np.any(np.isnan(img)):
				print("Fuck NAN")
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img)
			if img.max() == img.min():
				print(img)
			img = (img - img.min()) / (img.max() - img.min())
			return img.repeat(n_channels, 1, 1)
		return img_to_hist_accurate_blur
	elif method == "hist_accurate_blur_adv":
		def img_to_hist_accurate_blur(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img)
			img = (img - img.min()) / (img.max() - img.min())
			img = img + torch.randn_like(img) * 0.05
			img = (img - img.min()) / (img.max() - img.min())
			return img.repeat(n_channels, 1, 1)
		return img_to_hist_accurate_blur
	elif method == "hist_accurate_blur_tst":
		def img_to_hist_accurate_blur(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img)
			img = (img - img.min()) / (img.max() - img.min())
			img = transforms.functional.adjust_gamma(img, 0.6)
			img = (img - img.min()) / (img.max() - img.min())
			img = img.repeat(n_channels, 1, 1)
			return img
		return img_to_hist_accurate_blur
	elif method == "hist_accurate_blur_crop":
		def img_to_hist_accurate_blur(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img)
			img = transforms.CenterCrop(180)(img)
			img = (img - img.min()) / (img.max() - img.min())
			img = img.repeat(n_channels, 1, 1)
			return img
		return img_to_hist_accurate_blur
	elif method == "hist_accurate_crop":
		def img_to_hist_accurate_blur(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.CenterCrop(180)(img)
			img = (img - img.min()) / (img.max() - img.min())
			img = img.repeat(n_channels, 1, 1)
			return img
		return img_to_hist_accurate_blur
	elif method == "hist_accurate_blur_resize":
		def img_to_hist_accurate_blur(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img)
			img = transforms.CenterCrop(args.crop_size)(img)
			img = transforms.Resize(args.downsample_size)(img)
			img = torch.minimum(img, torch.tensor(args.trunc_depth))
			img = (img - img.min()) / (img.max() - img.min())
			img = img.repeat(n_channels, 1, 1)
			return img
		return img_to_hist_accurate_blur
	elif method == "hist_accurate_resize":
		def img_to_hist_accurate_blur(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.CenterCrop(args.crop_size)(img)
			img = transforms.Resize(args.downsample_size)(img)
			img = (img - img.min()) / (img.max() - img.min())
			img = img.repeat(n_channels, 1, 1)
			return img
		return img_to_hist_accurate_blur
	elif method == "hist_accurate_blur_resize_adj":
		def img_to_hist_accurate_blur(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img)
			img = transforms.CenterCrop(args.crop_size)(img)
			img = transforms.Resize(args.downsample_size)(img)
			img = torch.minimum(img, torch.tensor(args.trunc_depth))
			img = (img - img.min()) / (img.max() - img.min())
			img = img.repeat(n_channels, 1, 1)
			return img
		return img_to_hist_accurate_blur
	elif method == "hist_accurate_blur_jitter":
		blur = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)
		color_jitter = transforms.ColorJitter(brightness = 0.1, contrast = [0.7, 1.1], saturation = 0, hue = 0)
		def img_to_hist_accurate_blur_jitter(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = blur(img)
			img = (img - img.min()) / (img.max() - img.min())
			img = color_jitter(img)
			return img.repeat(n_channels, 1, 1)
		return img_to_hist_accurate_blur_jitter
	elif method == "small_01_blur":
		def fun(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img)
			img = (img - img.min()) / (img.max() - img.min())
			img = img.repeat(n_channels, 1, 1)
			img = transforms.Resize(112)(img)
			return img
		return fun
	elif method == "small_01":
		def fun(img):
			img = torch.tensor(img).float().unsqueeze(0)
			img = (img - img.min()) / (img.max() - img.min())
			img = img.repeat(n_channels, 1, 1)
			img = transforms.Resize(112)(img)
			return img
		return fun
	elif method == "hist_complex_blur":
		def img_to_hist_complex(img, bins = 30):
			orig_shape = img.shape
			img = torch.tensor(img).float().unsqueeze(0)
			img = transforms.GaussianBlur(args.blur_kernel, args.blur_sigma)(img).numpy().reshape(orig_shape)
			if np.allclose(img.max(), img.min()):
				img = np.zeros_like(img)
			else:
				img = (img - img.min()) / (img.max() - img.min())
			img_hist_indices = np.minimum(np.floor(img * bins).astype(int), bins - 1)
			img_hist_heights = np.histogram(img.ravel(), bins = bins, density = True)[0]
			hist_peak_idx = np.argmax(img_hist_heights)
			img_hist_heights = img_hist_heights / img_hist_heights[hist_peak_idx]
			#print("Prev: ", [round(x, 1) for x in img_hist_heights])
			for j in range(len(img_hist_heights)):
				i = len(img_hist_heights) - j - 1
				if i < hist_peak_idx:
					#img_hist_heights[i] += 1
					img_hist_heights[i] = 1
				if j > 0:
					img_hist_heights[i] = max(img_hist_heights[i], img_hist_heights[i + 1])
			#print("Succ: ", [round(x, 1) for x in img_hist_heights])
			labels = img_hist_heights[img_hist_indices]
			labels = labels.reshape(orig_shape)
			#mean, std = stats["hist_complex_mean"], stats["hist_complex_std"]
			#labels = (labels - mean) / std
			return torch.tensor(labels).float().unsqueeze(0).repeat(n_channels, 1, 1)
		return img_to_hist_complex
	
	elif method == "hist_complex":
		def img_to_hist_complex(img, bins = 30):
			orig_shape = img.shape
			if np.allclose(img.max(), img.min()):
				img = np.zeros_like(img)
			else:
				img = (img - img.min()) / (img.max() - img.min())
			img_hist_indices = np.minimum(np.floor(img * bins).astype(int), bins - 1)
			img_hist_heights = np.histogram(img.ravel(), bins = bins, density = True)[0]
			hist_peak_idx = np.argmax(img_hist_heights)
			img_hist_heights = img_hist_heights / img_hist_heights[hist_peak_idx]
			#print("Prev: ", [round(x, 1) for x in img_hist_heights])
			for j in range(len(img_hist_heights)):
				i = len(img_hist_heights) - j - 1
				if i < hist_peak_idx:
					#img_hist_heights[i] += 1
					img_hist_heights[i] = 1
				if j > 0:
					img_hist_heights[i] = max(img_hist_heights[i], img_hist_heights[i + 1])
			#print("Succ: ", [round(x, 1) for x in img_hist_heights])
			labels = img_hist_heights[img_hist_indices]
			labels = labels.reshape(orig_shape)
			#mean, std = stats["hist_complex_mean"], stats["hist_complex_std"]
			#labels = (labels - mean) / std
			return torch.tensor(labels).float().unsqueeze(0).repeat(n_channels, 1, 1)
		return img_to_hist_complex
	elif method == "hist_even_more_complex":
		assert n_channels == 2
		def img_to_hist_even_more_complex(img, bins = 40):
			orig_shape = img.shape
			if np.allclose(img.max(), img.min()):
				img = np.zeros_like(img)
			else:
				img = (img - img.min()) / (img.max() - img.min())
			jitter_sigma = 0.05
			if train:
				img = img + np.random.randn(*img.shape) * jitter_sigma
			img_hist_indices = np.minimum(np.floor(img * bins).astype(int), bins - 1)
			img_residual = img - img_hist_indices / bins
			img_hist_heights = np.histogram(img.ravel(), range = (0., 1.), bins = bins, density = True)[0]
			hist_peak_idx = np.argmax(img_hist_heights)
			img_hist_heights = img_hist_heights / img_hist_heights[hist_peak_idx]
			#print("Prev: ", [round(x, 1) for x in img_hist_heights])
			for j in range(len(img_hist_heights)):
				i = len(img_hist_heights) - j - 1
				if i < hist_peak_idx:
					#img_hist_heights[i] += 1
					img_hist_heights[i] = 1
				if j > 0:
					img_hist_heights[i] = max(img_hist_heights[i], img_hist_heights[i + 1])
			#print("Succ: ", [round(x, 1) for x in img_hist_heights])
			labels_l = img_hist_heights[img_hist_indices]
			img_hist_heights_extended = np.concatenate([img_hist_heights, np.array([img_hist_heights[-1]])], axis = 0)
			labels_r = img_hist_heights_extended[img_hist_indices + 1]
			final_labels = labels_l * img_residual + labels_r * (1 - img_residual)
			final_labels = final_labels.reshape(orig_shape)
			img_hist_indices = img_hist_indices.reshape(orig_shape)

			sigma = randu_gen(0.1, 2.3)()
			#centre_shift = randu_gen(0, 10)()
			#zoom = randu_gen(0.8, 1.2)()

			res_stack = []
			for t_channel in [final_labels, img_hist_indices]:
				if train:
					t_channel = ndimage.gaussian_filter(t_channel, sigma = sigma)
				#t_channel = ndimage.zoom()
				res_stack.append(t_channel)
			
			res = np.stack(res_stack, axis = 0)
			#mean, std = stats["hist_complex_mean"], stats["hist_complex_std"]
			#labels = (labels - mean) / std
			return torch.tensor(res).float()
		return img_to_hist_even_more_complex
	elif method == "success_augs":
		blur = transforms.GaussianBlur(21, 7)
		elastic = transforms.ElasticTransform(alpha = 50., sigma = 5.)
		persp = transforms.RandomPerspective(distortion_scale = 0.1, p = 0.5)
		crop_train = transforms.RandomResizedCrop(args.target_size, scale = (0.9, 1.0), ratio = (0.95, 1.05))
		resize = transforms.Resize(args.target_size)
		def fun(img):
			img = torch.tensor(img).unsqueeze(0)
			if training:
				img = blur(img)
				img = torch.minimum(img, torch.tensor(args.cap))
				if args.aug:
					img = elastic(img)
					img = persp(img)
					img = crop_train(img)
				else:
					img = resize(img)
			else:
				img = resize(img)
			img = (img - img.min()) / (img.max() - img.min())
			img = transforms.functional.pad(img, args.border_padding)
			return img.repeat(n_channels, 1, 1)
		return fun
	else:
		raise NotImplementedError

def get_pose_transforms(data_stats_path, hispose_noise, modality):
	stats = json.load(open(data_stats_path))
	pose_mean, pose_std = np.array(stats["pose_mean"]), np.array(stats["pose_std"])
	def trans_norm(x, true_pose):
		x = (x - pose_mean) / pose_std
		if not true_pose:
			x = x + np.random.randn(*x.shape) * hispose_noise
		return x
	trans = trans_norm
	inv_trans = lambda x : x * pose_std + pose_mean
	return trans, inv_trans

def get_pose_transforms_classification(all_cls):
	def inv_trans(x):
		if len(x.shape) == 2:
			return np.argmax(x, axis = -1)
		else:
			return x
	return lambda x, _ : x, inv_trans
