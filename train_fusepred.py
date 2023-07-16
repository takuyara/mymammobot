import torch
from torch import optim
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.nn_utils import get_loaders_loss_metrics, get_models

def train_val(model_atloc, model_fuser, model_fuse_predictor, dataloaders, optimiser, epochs, loss_fun, metric_template, device):
	model_atloc.eval()
	model_fuser.eval()
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model_fuse_predictor.train()
			else:
				model_fuse_predictor.eval()
			this_metric = metric_template.new_copy()
			for b_id, (imgs, poses) in enumerate(dataloaders[phase]):
				with torch.set_grad_enabled(phase == "train"):
					imgs, poses = imgs.to(device).float(), poses.to(device).float()
					#imgs_input, poses_input = imgs[ : , ]
					#b, s, c, w, h to b, c, s, w, h
					poses_history, poses_true = poses[ : , : -1, ...], poses[ : , -1, ...]
					bs, sqlen, C, W, H = tuple(imgs.shape)
					imgs = imgs.reshape(bs * sqlen, C, W, H)
					imgs_encode = model_atloc(imgs, get_encode = True)
					imgs_encode = imgs_encode.reshape(bs, sqlen, imgs_encode.size(1))
					history_encode, current_encode = imgs_encode[ : , : -1, : ], imgs_encode[ : , -1, : ]
					history_encode = model_fuser(history_encode, poses_history, get_encode = True).detach()
					current_encode = current_encode.detach()
					poses_pred = model_fuse_predictor(history_encode, current_encode)
					loss = loss_fun(poses_true, poses_pred)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				this_metric.add_batch(poses_true, poses_pred)
			print("Epoch {} {} done. Metrics: {}".format(i, phase, this_metric), flush = True)
			if phase == "val" and (i == 0 or this_metric.main_metric() < best_metric.main_metric()):
				best_metric, min_epoch, best_state_dict = this_metric, i, deepcopy(model_fuse_predictor.state_dict())
	print("Best metric {} occured at epoch {}".format(best_metric, min_epoch))
	return best_metric.loss(), min_epoch, best_state_dict

def main():
	args = get_args("atloc", "hisenc", "fusepred")
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args)
	model_atloc, model_fuser, model_fuse_predictor, device = get_models(args, "atloc+", "hisenc+", "fusepred")
	optimiser = optim.Adam(model_fuse_predictor.parameters(), lr = args.learning_rate)
	min_loss, min_epoch, best_state_dict = train_val(model_atloc, model_fuser, model_fuse_predictor, dataloaders, optimiser, args.epochs, loss_fun, metric_template, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"hisenc_fuse_predictor_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
