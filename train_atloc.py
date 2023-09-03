import torch
from torch import optim
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.nn_utils import get_loaders_loss_metrics, get_models

def train_val(model, dataloaders, optimiser, epochs, loss_fun, metric_template, device):
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model.train()
			else:
				model.eval()
			this_metric = metric_template.new_copy()
			for b_id, (imgs, poses) in enumerate(dataloaders[phase]):
				with torch.set_grad_enabled(phase == "train"):
					imgs, poses_true = imgs.to(device), poses.to(device)
					#print(imgs.shape, poses_true.shape)
					#b, s, c, w, h to b, c, s, w, h
					imgs = imgs.view(-1, *imgs.shape[-3 : ])
					#print(imgs.shape)
					poses_pred = model(imgs)
					if poses_true.dim() > 1:
						poses_pred = poses_pred.view(poses_true.shape)
					loss = loss_fun(poses_pred, poses_true)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				this_metric.add_batch(poses_pred, poses_true)
			print("Epoch {} {} done. Metrics: {}".format(i, phase, this_metric), flush = True)
			if phase == "val" and (i == 0 or this_metric.main_metric() < best_metric.main_metric()):
				best_metric, min_epoch, best_state_dict = this_metric, i, deepcopy(model.state_dict())
	print("Best metric {} occured at epoch {}".format(best_metric, min_epoch))
	return best_metric.loss(), min_epoch, best_state_dict

def main():
	args = get_args("atloc")
	train_set = "single_tc" if args.uses_tc else "single"
	val_set = "single_tc" if args.uses_tc and args.val_tc else "single"
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args, dset_names = [train_set, val_set])
	model, device = get_models(args, "atloc")
	optimiser = optim.Adam([p for p in model.parameters()] + [p for p in loss_fun.parameters()], lr = args.learning_rate)
	min_loss, min_epoch, best_state_dict = train_val(model, dataloaders, optimiser, args.epochs, loss_fun, metric_template, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"atloc_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
