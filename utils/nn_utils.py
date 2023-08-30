import os
import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader

from utils.file_utils import get_dir_list
from utils.reg_metrics import Metrics, BalancedL1Loss
from utils.preprocess import get_img_transform, get_pose_transforms

from datasets.cl_dataset import CLDataset, TestDataset
from datasets.single_dataset import SingleImageDataset
from models.atloc import AtLoc, PoseNet
from models.fuser import LSTMFuser
from models.fuse_predictor import MLPFusePredictor
from models.selector import MLPSelector

def l2_trans_loss(inp, tgt):
	return nn.MSELoss()(inp[..., : 3], tgt[..., : 3])

def get_loss_fun(args):
	if args.loss_fun == "l1":
		return nn.L1Loss()
	elif args.loss_fun == "l2":
		return nn.MSELoss()
	elif args.loss_fun == "l2_trans":
		return l2_trans_loss
	elif args.loss_fun == "balanced_l1":
		return BalancedL1Loss()
	else:
		raise NotImplementedError

def get_loaders_loss_metrics(args, test = False, test_set = False, single_img_set = False):
	input_modality = args.test_modality if test else "mesh"
	pose_trans, pose_inv_trans = get_pose_transforms(args.data_stats, args.hispose_noise, input_modality)
	if test:
		phase_split_path = [("test", args.test_split, args.test_preprocess, args.mesh_path if args.test_gen else None, args.test_rotatable)]
	else:
		phase_split_path = [("train", args.train_split, args.train_preprocess, args.mesh_path if args.train_gen else None, args.train_rotatable),
			("val", args.val_split, args.val_preprocess, args.mesh_path if args.val_gen else None, args.val_rotatable)]
	if not single_img_set:
		batch_size, ds_type = (1, TestDataset) if test and test_set else (args.batch_size, CLDataset)
		datasets = {phase : ds_type(args.base_dir, get_dir_list(split_path), args.length, args.spacing, args.img_size, mesh_path, args.skip_prev_frame,
			transform_img = get_img_transform(args.data_stats, preprocess, args.n_channels, phase == "train"), transform_pose = pose_trans) for phase, split_path, preprocess, mesh_path, __ in phase_split_path}
	else:
		batch_size = args.batch_size
		datasets = {phase : SingleImageDataset(args.base_dir, get_dir_list(split_path), args.img_size, mesh_path, rotatable = rotate,
			transform_img = get_img_transform(args.data_stats, preprocess, args.n_channels, phase == "train"), transform_pose = pose_trans) for phase, split_path, preprocess, mesh_path, rotate in phase_split_path}

	dataloaders = {phase : DataLoader(ds, batch_size = batch_size,
		num_workers = args.num_workers, shuffle = phase == "train") for phase, ds in datasets.items()}
	if test:
		dataloaders = dataloaders["test"]
	loss_fun = get_loss_fun(args)
	return dataloaders, loss_fun, Metrics(loss_fun, pose_inv_trans, args.model_sel_metric, args.model_sel_rot_coef)

def get_models(args, *names):
	device = torch.device(args.device)
	res_models = []
	for t_name in names:
		if t_name.startswith("atloc"):
			if args.atloc_base == "resnet18":
				base = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
			elif args.atloc_base == "resnet34":
				base = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
			elif args.atloc_base == "resnet50":
				base = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
			if args.uses_posenet:
				model = PoseNet(base, droprate = args.dropout, n_channels = args.n_channels).to(device)
			else:
				model = AtLoc(base, droprate = args.dropout, feat_dim = args.img_encode_dim, n_channels = args.n_channels).to(device)
			if t_name.endswith("+"):
				model.load_state_dict(torch.load(os.path.join(args.save_path, args.atloc_path)))
		elif t_name.startswith("hisenc"):
			model = LSTMFuser(args.length, args.img_encode_dim, args.output_dim, args.hidden_size, args.num_layers,
				args.mlp_hisenc_out, args.output_dim, args.dropout, args.bidirectional, args.img_his_only).to(device)
			if t_name.endswith("+"):
				model.load_state_dict(torch.load(os.path.join(args.save_path, args.hisenc_path)))
		elif t_name.startswith("fusepred"):
			model = MLPFusePredictor(args.mlp_hisenc_out[-1], args.img_encode_dim, args.mlp_branch_his,
				args.mlp_branch_cur, args.mlp_fusepred_out, args.output_dim, args.dropout, args.fuse_mode).to(device)
			if t_name.endswith("+"):
				model.load_state_dict(torch.load(os.path.join(args.save_path, args.fusepred_path)))
		elif t_name.startswith("finalsel"):
			model = MLPSelector(args.img_encode_dim, args.mlp_weighting, args.dropout, 3, args.output_dim, args.negative_weights).to(device)
			if t_name.endswith("+"):
				model.load_state_dict(torch.load(os.path.join(args.save_path, args.finalsel_path)))
		else:
			raise NotImplementedError
		res_models.append(model)
	return tuple(res_models + [device])
