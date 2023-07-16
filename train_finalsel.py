import torch
from torch import optim
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.pose_utils import Metrics
from utils.nn_utils import get_loss_fun, get_loaders, get_models

def train_val(model_atloc, model_fuser, model_fuse_predictor, model_sel, dataloaders, optimiser, epochs, loss_fun, pose_inv_trans, device):
	model_atloc.eval()
	model_fuser.eval()
	model_fuse_predictor.eval()
	min_loss = 1e100
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model_sel.train()
			else:
				model_sel.eval()
			this_metric = Metrics(loss_fun, pose_inv_trans)
			for b_id, (imgs, poses) in enumerate(dataloaders[phase]):
				with torch.set_grad_enabled(phase == "train"):
					imgs, poses = imgs.to(device).float(), poses.to(device).float()
					#imgs_input, poses_input = imgs[ : , ]
					#b, s, c, w, h to b, c, s, w, h
					poses_history, poses_true = poses[ : , : -1, ...], poses[ : , -1, ...]
					bs, sqlen, C, W, H = tuple(imgs.shape)
					imgs = imgs.reshape(bs * sqlen, C, W, H)
					imgs_encode, atloc_pred = model_atloc(imgs, get_encode = True, return_both = True)
					imgs_encode = imgs_encode.reshape(bs, sqlen, imgs_encode.size(1))
					atloc_pred = atloc_pred.reshape(bs, sqlen, atloc_pred.size(1))[ : , -1, : ]
					history_encode, current_encode = imgs_encode[ : , : -1, : ], imgs_encode[ : , -1, : ]
					history_encode, fuser_pred = model_fuser(history_encode, poses_history, get_encode = True, return_both = True)
					fusepred_pred = model_fuse_predictor(history_encode, current_encode)
					current_encode, atloc_pred, fuser_pred, fusepred_pred = current_encode.detach(), atloc_pred.detach(), fuser_pred.detach(), fusepred_pred.detach()
					sel_pred = model_sel(current_encode, atloc_pred, fuser_pred, fusepred_pred)
					loss = loss_fun(poses_true, sel_pred)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				this_metric.add_batch(poses_true, sel_pred)
			print("Epoch {} {} done. Metrics: {}".format(i, phase, this_metric), flush = True)
			if phase == "val" and this_metric.loss() < min_loss:
				min_loss, min_epoch, best_state_dict = this_metric.loss(), i, deepcopy(model_sel.state_dict())
	print("Min loss {:.5f} occured at epoch {}".format(min_loss, min_epoch))
	return min_loss, min_epoch, best_state_dict

def main():
	args = get_args("atloc", "hisenc", "fusepred", "finalsel")
	dataloaders, pose_inv_trans = get_loaders(args)
	model_atloc, model_fuser, model_fuse_predictor, model_sel, device = get_models(args, "atloc+", "hisenc+", "fusepred+", "finalsel")
	optimiser = optim.Adam(model_sel.parameters(), lr = args.learning_rate)
	min_loss, min_epoch, best_state_dict = train_val(model_atloc, model_fuser, model_fuse_predictor, model_sel, dataloaders, optimiser, args.epochs, get_loss_fun(args), pose_inv_trans, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"selector_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
