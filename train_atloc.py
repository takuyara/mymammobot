import torch
from torch import optim
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.pose_utils import Metrics
from utils.losses import get_loss_fun, get_loaders, get_models

def train_val(model, dataloaders, optimiser, epochs, loss_fun, pose_inv_trans, device):
	min_loss = 1e100
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model.train()
			else:
				model.eval()
			this_metric = Metrics(loss_fun, pose_inv_trans)
			for b_id, (imgs, poses) in enumerate(dataloaders[phase]):
				with torch.set_grad_enabled(phase == "train"):
					imgs, poses = imgs.to(device).float(), poses.to(device).float()
					#b, s, c, w, h to b, c, s, w, h
					imgs, poses_true = imgs[ : , -1, ...], poses[ : , -1, ...]
					poses_pred = model(imgs)
					loss = loss_fun(poses_true, poses_pred)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				this_metric.add_batch(poses_true, poses_pred)
			print("Epoch {} {} done. Metrics: {}".format(i, phase, this_metric), flush = True)
			if phase == "val" and this_metric.loss() < min_loss:
				min_loss, min_epoch, best_state_dict = this_metric.loss(), i, deepcopy(model.state_dict())
	print("Min loss {:.5f} occured at epoch {}".format(min_loss, min_epoch))
	return min_loss, min_epoch, best_state_dict

def main():
	args = get_args("atloc")
	dataloaders, pose_inv_trans = get_loaders(args)
	model, device = get_models(args, "atloc")
	optimiser = optim.Adam(model.parameters(), lr = args.learning_rate)
	min_loss, min_epoch, best_state_dict = train_val(model, dataloaders, optimiser, args.epochs, get_loss_fun(args), pose_inv_trans, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"atloc_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
