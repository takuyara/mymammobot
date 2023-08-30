import torch
from torch import optim
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.nn_utils import get_loaders_loss_metrics, get_models

def train_val(model, dataloaders, loss_fun, metric_template, device):
	model.eval()

	for (imgs_v, poses_v), (imgs_r, poses_r) in zip(dataloaders["virtual"], dataloaders["real"]):
		with torch.no_grad():
			assert torch.abs(poses_v - poses_r).max() < 1e-4
			this_metric_v = metric_template.new_copy()
			this_metric_r = metric_template.new_copy()
			imgs_v, poses_true_v = imgs_v.to(device).float(), poses_v.to(device).float()
			imgs_r, poses_true_r = imgs_r.to(device).float(), poses_r.to(device).float()
			#b, s, c, w, h to b, c, s, w, h
			poses_pred_v = model(imgs_v)
			poses_pred_r = model(imgs_r)
			loss_v = loss_fun(poses_true_v, poses_pred_v)
			loss_r = loss_fun(poses_true_r, poses_pred_r)
			this_metric_v.add_batch(poses_true_v, poses_pred_v)
			this_metric_r.add_batch(poses_true_r, poses_pred_r)
			print("Virtual: ", this_metric_v)
			print("Real: ", this_metric_r)
			errs = torch.abs(imgs_v - imgs_r)
			print("Dom error: {:.4f} {:.4f} {:.4f}".format(torch.mean(errs), torch.min(errs), torch.max(errs)))

def main():
	args = get_args("atloc", "hisenc")
	ald = {}
	args.batch_size = 1
	args.val_gen = False
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args, single_img_set = True)
	ald["real"] = dataloaders["val"]
	args.val_gen = True
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args, single_img_set = True)
	ald["virtual"] = dataloaders["val"]
	model, device = get_models(args, "atloc+")
	train_val(model, ald, loss_fun, metric_template, device)

if __name__ == '__main__':
	main()
