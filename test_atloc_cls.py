import torch
from torch import optim
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

import pyvista as pv

from utils.arguments import get_args
from utils.nn_utils import get_loaders_loss_metrics, get_models

from sklearn.metrics import confusion_matrix

from torchvision import transforms

def train_val(p, model, dataloaders, loss_fun, metric_template, device):
	model.eval()

	transes, err_means, err_maxes = [], [], []
	all_points_true, all_points_pred = [], []
	trans_err_v, trans_err_r = [], []
	both_bad, both_good, only_r_bad, only_v_bad = [], [], [], []
	r_preds, v_preds = [], []

	bad_thres = 15

	true_lbs, pred_lbs = [], []

	wrong_pts, right_pts = [[], [], []], []

	for (imgs_v, poses_v), (imgs_r, poses_r) in zip(dataloaders["virtual"], dataloaders["real"]):
		with torch.no_grad():
			#assert torch.abs(poses_v - poses_r).max() < 1e-4
			this_metric_v = metric_template.new_copy()
			this_metric_r = metric_template.new_copy()

			imgs_r, poses_true_r = imgs_r.to(device), poses_r.to(device)

			poses_pred_r = model(imgs_r)
			actual_poses = metric_template.inv_trans(poses_v.numpy()).reshape(-1, 6)[ : , : 3]
			pred_lbs_ = torch.argmax(poses_pred_r, dim = -1)
			imgs_r = imgs_r.cpu().numpy()

			for i in range(len(actual_poses)):
				true_lb, pred_lb, this_actual_pose = poses_true_r[i].item(), pred_lbs_[i].item(), actual_poses[i, ...]
				true_lbs.append(true_lb)
				pred_lbs.append(pred_lb)
				if true_lb != pred_lb:
					wrong_pts[true_lb].append(this_actual_pose)
				else:
					right_pts.append(this_actual_pose)
				"""
				if true_lb in [1, 2] and pred_lb == true_lb:
					plt.imshow(imgs_r[i].reshape(224, 224))
					plt.title(pred_lb)
					plt.show()
				"""
				if true_lb == 0 and true_lb != pred_lb:
					print(poses_pred_r[i, ...])

	
	#p.add_points(np.stack(all_points_true, axis = 0), render_points_as_spheres = True, point_size = 5, color = "red")
	#p.add_points(np.stack(all_points_pred, axis = 0), render_points_as_spheres = True, point_size = 5, color = "blue")
	
	#p.add_points(np.stack(both_good, axis = 0), render_points_as_spheres = True, point_size = 10, color = "green")
	#p.add_points(np.stack(both_bad, axis = 0), render_points_as_spheres = True, point_size = 10, color = "black")
	for points, colour in zip(wrong_pts, ["red", "blue", "green"]):
		p.add_points(np.stack(points, axis = 0), render_points_as_spheres = True, point_size = 10, color = colour)
	p.add_points(np.stack(right_pts, axis = 0), render_points_as_spheres = True, point_size = 10, color = "black")
	#p.add_points(np.stack(r_preds, axis = 0), render_points_as_spheres = True, point_size = 10, color = "blue")
	
	mat = confusion_matrix(true_lbs, pred_lbs)
	for i in range(len(mat)):
		print(f"true={i}", " ".join(["{}".format(mat[i, j]) for j in range(len(mat))]))
	

	
	"""
	for true_pose, real_pred in zip(only_r_bad, v_preds):
		ln = pv.Line(true_pose, real_pred)
		p.add_mesh(ln, color = "red", line_width = 3)
		p.add_mesh(pv.Arrow(real_pred, real_pred - true_pose, tip_radius = 0.3, tip_length = 0.75), color = "red")
	"""

	


	p.show()
	exit()

	print(np.mean(trans_err_v), np.mean(trans_err_r))

	plt.subplot(1, 2, 1)
	plt.hist(trans_err_v, bins = 50)
	plt.title("Mesh Val Trans Errors")
	plt.xlabel("Trans Error")
	plt.ylabel("Counts")

	plt.subplot(1, 2, 2)
	plt.hist(trans_err_r, bins = 50)
	plt.title("SFS Test Trans Errors")
	plt.xlabel("Trans Error")
	plt.ylabel("Counts")
	plt.show()

	plt.subplot(1, 2, 1)
	plt.scatter(err_means, transes)
	plt.title("MAE vs SFS Error - Mesh Error")
	plt.xlabel("MAE in Domain Transfer")
	plt.ylabel("How Much Worse is SFS Pred")
	plt.subplot(1, 2, 2)
	plt.scatter(err_maxes, transes)
	plt.title("Max AE vs SFS Error - Mesh Error")
	plt.xlabel("Max AE in Domain Transfer")
	plt.ylabel("How Much Worse is SFS Pred")
	plt.show()

def main():
	args = get_args("atloc", "hisenc")
	p = pv.Plotter()
	p.add_mesh(pv.read(args.mesh_path), opacity = 0.5)

	ald = {}
	args.batch_size = 128
	args.val_gen = False
	args.val_preprocess = "hist_accurate_resize"
	args.cls = True
	args.train_split = "train"
	args.val_split = "val"
	args.base_dir = "./"
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args, dset_names = ["preload", "preload"])
	ald["real"] = dataloaders["val"]
	"""
	args.val_gen = True
	args.val_preprocess = "hist_accurate_blur"
	"""
	args.cls = False
	args.val_split = "real_confirmed.txt"
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args, dset_names = ["preload", "single"])
	ald["virtual"] = dataloaders["val"]
	model, device = get_models(args, "atloc+")
	train_val(p, model, ald, loss_fun, metric_template, device)

if __name__ == '__main__':
	main()
