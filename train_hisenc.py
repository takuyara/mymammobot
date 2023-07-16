import torch
from torch import optim
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.nn_utils import get_loaders_loss_metrics, get_models

def train_val(model_atloc, model_fuser, dataloaders, optimiser, epochs, loss_fun, metric_template, device):
	model_atloc.eval()
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model_fuser.train()
			else:
				model_fuser.eval()
			this_metric = metric_template.new_copy()
			for b_id, (imgs, poses) in enumerate(dataloaders[phase]):
				with torch.set_grad_enabled(phase == "train"):
					imgs, poses = imgs.to(device).float(), poses.to(device).float()
					#imgs_input, poses_input = imgs[ : , ]
					#b, s, c, w, h to b, c, s, w, h
					imgs_history, poses_history = imgs[ : , : -1, ...], poses[ : , : -1, ...]
					poses_true = poses[ : , -1, ...]
					bs, sqlen, C, W, H = tuple(imgs_history.shape)
					imgs_history = imgs_history.reshape(bs * sqlen, C, W, H)
					imgs_history = model_atloc(imgs_history, get_encode = True).detach()
					imgs_history = imgs_history.reshape(bs, sqlen, imgs_history.size(1))
					poses_pred = model_fuser(imgs_history, poses_history)
					loss = loss_fun(poses_true, poses_pred)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				this_metric.add_batch(poses_true, poses_pred)
			print("Epoch {} {} done. Metrics: {}".format(i, phase, this_metric), flush = True)
			if phase == "val" and (i == 0 or this_metric.main_metric() < best_metric.main_metric()):
				best_metric, min_epoch, best_state_dict = this_metric, i, deepcopy(model_fuser.state_dict())
	print("Best metric {} occured at epoch {}".format(best_metric, min_epoch))
	return best_metric.loss(), min_epoch, best_state_dict

def main():
	args = get_args("atloc", "hisenc")
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args)
	model_atloc, model_fuser, device = get_models(args, "atloc+", "hisenc")
	optimiser = optim.Adam(model_fuser.parameters(), lr = args.learning_rate)
	min_loss, min_epoch, best_state_dict = train_val(model_atloc, model_fuser, dataloaders, optimiser, args.epochs, loss_fun, metric_template, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"hisenc_fuser_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
