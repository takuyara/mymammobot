import torch
from torch import optim
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

import pyvista as pv

from utils.arguments import get_args
from utils.nn_utils import get_loaders_loss_metrics, get_models

from torchvision import transforms

def train_val(p, model, dataloaders, loss_fun, metric_template, device):
	model.eval()

	transes, err_means, err_maxes = [], [], []
	all_points_true, all_points_pred = [], []
	trans_err_v, trans_err_r = [], []

	for (imgs_v, poses_v), (imgs_r, poses_r) in zip(dataloaders["virtual"], dataloaders["real"]):
		with torch.no_grad():
			assert torch.abs(poses_v - poses_r).max() < 1e-4
			this_metric_v = metric_template.new_copy()
			this_metric_r = metric_template.new_copy()

			imgs_v, poses_true_v = imgs_v.to(device).float(), poses_v.to(device).float()
			imgs_r, poses_true_r = imgs_r.to(device).float(), poses_r.to(device).float()

			#imgs_r = transforms.Compose([transforms.CenterCrop(190), transforms.Resize(224)])(imgs_r)
			#b, s, c, w, h to b, c, s, w, h
			poses_pred_v = model(imgs_v)
			poses_pred_r = model(imgs_r)
			loss_v = loss_fun(poses_true_v, poses_pred_v)
			loss_r = loss_fun(poses_true_r, poses_pred_r)
			this_metric_v.add_batch(poses_true_v, poses_pred_v)
			this_metric_r.add_batch(poses_true_r, poses_pred_r)
			#print("Virtual: ", this_metric_v)
			#print("Real: ", this_metric_r)
			errs = torch.abs(imgs_v - imgs_r)
			#print("Dom error: {:.4f} {:.4f} {:.4f}".format(torch.mean(errs), torch.min(errs), torch.max(errs)))

			dlt_trans = this_metric_r.get_dict()["translation_error"] - this_metric_v.get_dict()["translation_error"]
			trans_err_v.append(this_metric_v.get_dict()["translation_error"])
			trans_err_r.append(this_metric_r.get_dict()["translation_error"])
			transes.append(dlt_trans)
			err_means.append(torch.mean(errs).item())
			err_maxes.append(torch.max(errs).item())

			if dlt_trans > 30:
				all_points_true.append(poses_true_v.cpu().numpy().reshape(6)[ : 3])
				all_points_pred.append(poses_pred_r.cpu().numpy().reshape(6)[ : 3])
	
	"""
	p.add_points(np.stack(all_points_true, axis = 0), render_points_as_spheres = True, point_size = 5, color = "red")
	p.add_points(np.stack(all_points_pred, axis = 0), render_points_as_spheres = True, point_size = 5, color = "blue")
	p.show()
	"""

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
	p.add_mesh(pv.read(args.mesh_path))

	ald = {}
	args.batch_size = 1
	args.val_gen = False
	args.val_preprocess = "hist_accurate_blur"
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args, single_img_set = True)
	ald["real"] = dataloaders["val"]
	args.val_gen = True
	args.val_preprocess = "hist_accurate"
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args, single_img_set = True)
	ald["virtual"] = dataloaders["val"]
	model, device = get_models(args, "atloc+")
	train_val(p, model, ald, loss_fun, metric_template, device)

if __name__ == '__main__':
	main()
