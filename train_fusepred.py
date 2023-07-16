import torch
from torch import optim
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.pose_utils import Metrics
from utils.nn_utils import get_loss_fun, get_loaders, get_models

def train_val(model_atloc, model_fuser, model_fuse_predictor, dataloaders, optimiser, epochs, loss_fun, pose_inv_trans, device):
	model_atloc.eval()
	model_fuser.eval()
	min_loss = 1e100
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model_fuse_predictor.train()
			else:
				model_fuse_predictor.eval()
			this_metric = Metrics(loss_fun, pose_inv_trans)
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
			if phase == "val" and this_metric.loss() < min_loss:
				min_loss, min_epoch, best_state_dict = this_metric.loss(), i, deepcopy(model_fuse_predictor.state_dict())
	print("Min loss {:.5f} occured at epoch {}".format(min_loss, min_epoch))
	return min_loss, min_epoch, best_state_dict

def main():
	args = get_args("atloc", "hisenc", "fusepred")
	dataloaders, pose_inv_trans = get_loaders(args)
	model_atloc, model_fuser, model_fuse_predictor, device = get_models(args, "atloc+", "hisenc+", "fusepred")
	optimiser = optim.Adam(model_fuse_predictor.parameters(), lr = args.learning_rate)
	min_loss, min_epoch, best_state_dict = train_val(model_atloc, model_fuser, model_fuse_predictor, dataloaders, optimiser, args.epochs, get_loss_fun(args), pose_inv_trans, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"hisenc_fuse_predictor_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
