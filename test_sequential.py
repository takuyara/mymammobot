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
from datasets.cl_dataset import TestDataset
from models.atloc import AtLoc
from models.fuser import LSTMFuser
from models.fuse_predictor import MLPFusePredictor
from models.selector import MLPSelector

def train_val(model_atloc, model_fuser, model_fuse_predictor, model_sel, dataloader, pose_inv_trans, val_models, device):
	model_atloc.eval()
	model_fuser.eval()
	model_fuse_predictor.eval()
	model_sel.eval()
	atloc_metric = Metrics(pose_inv_trans)
	hisenc_metric = Metrics(pose_inv_trans)
	fusepred_metric = Metrics(pose_inv_trans)
	finalsel_metric = Metrics(pose_inv_trans)
	atloc_poses = []
	hisenc_poses = []
	fusepred_poses = []
	finalsel_poses = []
	for b_id, (imgs, poses_true, his_indices, pred_id, use_hisenc) in enumerate(dataloader):
		with torch.no_grad():
			if val_models < 1:
				continue
			imgs, poses_true = imgs.to(device).float(), poses_true.to(device).float()
			his_indices, pred_id, use_hisenc = his_indices.to(device).int(), int(pred_id.item()), int(use_hisenc.item())
			#print(imgs.shape, poses_true.shape, his_indices.shape, pred_id, use_hisenc)
			#b, s, c, w, h 
			bs, sqlen, C, W, H = tuple(imgs.shape)
			imgs = imgs.reshape(bs * sqlen, C, W, H)
			imgs_encode, atloc_pred = model_atloc(imgs, get_encode = True, return_both = True)
			imgs_encode = imgs_encode.reshape(bs, sqlen, imgs_encode.size(1))
			history_imgs_encode, current_img_encode = imgs_encode[ : , : -1, : ], imgs_encode[ : , -1, : ]
			atloc_pred = atloc_pred.reshape(bs, sqlen, atloc_pred.size(1))[ : , -1, : ]
			atloc_metric.add_batch(atloc_pred, poses_true)
			assert len(atloc_poses) == pred_id
			atloc_poses.append(atloc_pred)
			if val_models < 2:
				continue

			
			def get_hisenc_results(his_poses):
				poses_history = torch.stack([his_poses[idx].flatten() for idx in his_indices.flatten()]).unsqueeze(0)
				history_encode, hisenc_pred = model_fuser(history_imgs_encode, poses_history, get_encode = True, return_both = True)
				return history_encode, hisenc_pred
			
			if use_hisenc == 1:
				hisenc_pred = get_hisenc_results(hisenc_poses)[1]
			else:
				hisenc_pred = atloc_pred.detach().clone()
			hisenc_metric.add_batch(hisenc_pred, poses_true)
			assert len(hisenc_poses) == pred_id
			hisenc_poses.append(hisenc_pred)
			if val_models < 3:
				continue

			if use_hisenc == 1:
				history_encode = get_hisenc_results(fusepred_poses)[0]
				fusepred_pred = model_fuse_predictor(history_encode, current_img_encode)
			else:
				fusepred_pred = atloc_pred.detach().clone()
			fusepred_metric.add_batch(fusepred_pred, poses_true)
			assert len(fusepred_poses) == pred_id
			fusepred_poses.append(fusepred_pred)
			if val_models < 4:
				continue

			if use_hisenc == 1:
				history_encode, hisenc_pred = get_hisenc_results(finalsel_poses)
				fusepred_pred = model_fuse_predictor(history_encode, current_img_encode)
				sel_pred = model_sel(current_img_encode, atloc_pred, hisenc_pred, fusepred_pred)
			else:
				sel_pred = atloc_pred.detach().clone()
			finalsel_metric.add_batch(sel_pred, poses_true)
			assert len(finalsel_poses) == pred_id
			finalsel_poses.append(sel_pred)

	all_val = [("AtLoc", atloc_metric),("HisEnc", hisenc_metric), ("FusePred", fusepred_metric), ("FinalSel", finalsel_metric)]
	return all_val[ : val_models]

def main():
	args = get_args("atloc", "hisenc", "fusepred", "finalsel", "test")
	pose_trans, pose_inv_trans = get_pose_transforms(args)
	dataset = TestDataset(args.base_dir, get_dir_list(args.test_split), args.length, args.spacing, get_img_transform(args), pose_trans)
	dataloader = DataLoader(dataset, batch_size = 1, num_workers = args.num_workers, shuffle = False)
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
	model_sel = MLPSelector(args.img_encode_dim, args.mlp_weighting, args.dropout, 3, args.output_dim, args.negative_weights).to(device)
	if args.finalsel_path is not None and val_models == 3:
		model_sel.load_state_dict(torch.load(args.finalsel_path))
		val_models = 4
	val_results = train_val(model_atloc, model_fuser, model_fuse_predictor, model_sel, dataloader, pose_inv_trans, val_models, device)
	for model_name, results in val_results:
		print(model_name, results)

if __name__ == '__main__':
	main()
