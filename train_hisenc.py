import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.file_utils import get_dir_list
from utils.preprocess import get_img_transform, get_pose_transforms
from utils.pose_utils import Metrics
from datasets.cl_dataset import CLDataset
from models.atloc import AtLoc
from models.fuser import LSTMFuser

def train_val(model_atloc, model_fuser, dataloaders, optimiser, epochs, pose_inv_trans, device):
	model_atloc.eval()
	min_loss = 1e100
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model_fuser.train()
			else:
				model_fuser.eval()
			this_metric = Metrics(pose_inv_trans)
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
					loss = nn.MSELoss()(poses_true, poses_pred)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				this_metric.add_batch(poses_true, poses_pred)
			print("Epoch {} {} done. Metrics: {}".format(i, phase, this_metric), flush = True)
			if phase == "val" and this_metric.loss() < min_loss:
				min_loss, min_epoch, best_state_dict = this_metric.loss(), i, deepcopy(model_fuser.state_dict())
	print("Min loss {:.5f} occured at epoch {}".format(min_loss, min_epoch))
	return min_loss, min_epoch, best_state_dict

def main():
	args = get_args("atloc", "hisenc")
	pose_trans, pose_inv_trans = get_pose_transforms(args)
	datasets = {phase : CLDataset(args.base_dir, get_dir_list(split_path), args.length, args.spacing,
		get_img_transform(args), pose_trans) for phase, split_path in [("train", args.train_split), ("val", args.val_split)]}
	dataloaders = {phase : DataLoader(ds, batch_size = args.batch_size,
		num_workers = args.num_workers, shuffle = phase == "train") for phase, ds in datasets.items()}
	device = torch.device(args.device)
	model_atloc = AtLoc(models.resnet34(pretrained = False), droprate = args.dropout, feat_dim = args.img_encode_dim).to(device)
	model_atloc.load_state_dict(torch.load(args.atloc_path))
	model_fuser = LSTMFuser(args.length, args.img_encode_dim, args.output_dim, args.hidden_size, args.num_layers, args.mlp_hisenc_out, args.output_dim, args.dropout).to(device)
	optimiser = optim.Adam(model_fuser.parameters(), lr = args.learning_rate)
	min_loss, min_epoch, best_state_dict = train_val(model_atloc, model_fuser, dataloaders, optimiser, args.epochs, pose_inv_trans, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"hisenc_fuser_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
