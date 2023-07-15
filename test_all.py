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
from models.fuse_predictor import MLPFusePredictor
from models.selector import MLPSelector

def train_val(model_atloc, model_fuser, model_fuse_predictor, model_sel, dataloaders, pose_inv_trans, val_models, device):
	model_atloc.eval()
	model_fuser.eval()
	model_fuse_predictor.eval()
	model_sel.eval()
	atloc_metric = Metrics(pose_inv_trans)
	hisenc_metric = Metrics(pose_inv_trans)
	fusepred_metric = Metrics(pose_inv_trans)
	finalsel_metric = Metrics(pose_inv_trans)
	for b_id, (imgs, poses) in enumerate(dataloader):
		with torch.no_grad():
			if val_models < 1:
				continue
			imgs, poses = imgs.to(device).float(), poses.to(device).float()
			#imgs_input, poses_input = imgs[ : , ]
			#b, s, c, w, h to b, c, s, w, h
			poses_history, poses_true = poses[ : , : -1, ...], poses[ : , -1, ...]
			bs, sqlen, C, W, H = tuple(imgs.shape)
			imgs = imgs.reshape(bs * sqlen, C, W, H)
			imgs_encode, atloc_pred = model_atloc(imgs, get_encode = True, return_both = True)
			atloc_metric.add_batch(atloc_pred, poses_true)
			if val_models < 2:
				continue
			imgs_encode = imgs_encode.reshape(bs, sqlen, imgs_encode.size(1))
			atloc_pred = atloc_pred.reshape(bs, sqlen, atloc_pred.size(1))[ : , -1, : ]
			history_encode, current_encode = imgs_encode[ : , : -1, : ], imgs_encode[ : , -1, : ]
			history_encode, fuser_pred = model_fuser(history_encode, poses_history, get_encode = True, return_both = True)
			hisenc_metric.add_batch(fuser_pred, poses_true)
			if val_models < 3:
				continue
			fusepred_pred = model_fuse_predictor(history_encode, current_encode)
			fusepred_metric.add_batch(fusepred_pred, poses_true)
			if val_models < 4:
				continue
			current_encode, atloc_pred, fuser_pred, fusepred_pred = current_encode.detach(), atloc_pred.detach(), fuser_pred.detach(), fusepred_pred.detach()
			sel_pred = model_sel(current_encode, atloc_pred, fuser_pred, fusepred_pred)
			finalsel_metric.add_batch(sel_pred, poses_true)
	all_val = [("AtLoc", atloc_metric),("HisEnc", hisenc_metric), ("FusePred", fusepred_metric), ("FinalSel", finalsel_metric)]
	return all_val[ : val_models]

def main():
	args = get_args("atloc", "hisenc", "fusepred", "finalsel", "test")
	pose_trans, pose_inv_trans = get_pose_transforms(args)
	dataset = CLDataset(args.base_dir, get_dir_list(args.test_split), args.length, args.spacing, get_img_transform(args), pose_trans)
	dataloader = DataLoader(ds, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
	device = torch.device(args.device)
	val_models = 0
	model_atloc = AtLoc(models.resnet34(pretrained = False), droprate = args.dropout, feat_dim = args.img_encode_dim).to(device)
	if args.atloc_path is not None:
		model_atloc.load_state_dict(torch.load(args.atloc_path))
		val_models = 1
	model_fuser = LSTMFuser(args.length, args.img_encode_dim, args.output_dim, args.hidden_size, args.num_layers, args.mlp_hisenc_out, args.output_dim, args.dropout).to(device)
	if args.hisenc_path is not None and val_models == 1:
		model_fuser.load_state_dict(torch.load(args.hisenc_path))
		val_models = 2
	model_fuse_predictor = MLPFusePredictor(args.mlp_hisenc_out[-1], args.img_encode_dim, args.mlp_branch_his, args.mlp_branch_cur, args.mlp_fusepred_out, args.output_dim, args.dropout, args.fuse_mode).to(device)
	if args.fusepred_path is not None and val_models == 2:
		model_fuse_predictor.load_state_dict(torch.load(args.fusepred_path))
		val_models = 3
	model_sel = MLPSelector(args.img_encode_dim, args.mlp_weighting, args.dropout, 3).to(device)
	if args.finalsel_path is not None and val_models == 3:
		model_sel.load_state_dict(torch.load(args.finalsel_path))
		val_models = 4
	val_results = train_val(model_atloc, model_fuser, model_fuse_predictor, model_sel, dataloader, pose_inv_trans, val_models, device)
	for model_name, results in val_results:
		print(model_name, results)

if __name__ == '__main__':
	main()

