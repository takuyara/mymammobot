import os
import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader

from utils.cl_utils import load_all_cls_npy
from utils.file_utils import get_dir_list
from utils.reg_metrics import Metrics_Reg, Metrics_Cls, BalancedL1Loss, TransL2Loss, TCLoss
from utils.preprocess import get_img_transform, get_pose_transforms, get_pose_transforms_classification

from datasets.cl_dataset import CLDataset, TestDataset
from datasets.single_dataset import SingleImageDataset
from datasets.single_tcset import SingleTCDataset
from models.atloc import AtLoc, PoseNet
from models.fuser import LSTMFuser
from models.fuse_predictor import MLPFusePredictor
from models.selector import MLPSelector

def get_loss_fun(args):
	if args.cls:
		bloss = nn.CrossEntropyLoss()
		assert not args.uses_tc
	elif args.loss_fun == "l1":
		bloss = nn.L1Loss()
	elif args.loss_fun == "l2":
		bloss = nn.MSELoss()
	elif args.loss_fun == "l2_trans":
		bloss = TransL2Loss()
	elif args.loss_fun == "balanced_l1":
		bloss = BalancedL1Loss()
	else:
		raise NotImplementedError
	if args.uses_tc:
		bloss = TCLoss(bloss, relative_coef = args.relative_coef)
	return bloss

def get_loaders_loss_metrics(args, test = False, dset_names = "single"):
	input_modality = args.test_modality if test else "mesh"
	if args.cls:
		all_cls = load_all_cls_npy(args.seg_cl_path)
		pose_trans, pose_inv_trans = get_pose_transforms_classification(all_cls)
		args.output_dim = len(all_cls)
		metric_name = Metrics_Cls
	else:
		pose_trans, pose_inv_trans = get_pose_transforms(args.data_stats, args.hispose_noise, input_modality)
		metric_name = Metrics_Reg
		all_cls = None

	if test:
		phase_split_path = [("test", args.test_split, args.test_preprocess, args.mesh_path if args.test_gen else None)]
	else:
		phase_split_path = [("train", args.train_split, args.train_preprocess, args.mesh_path if args.train_gen else None),
			("val", args.val_split, args.val_preprocess, args.mesh_path if args.val_gen else None)]

	if type(dset_names) != list:
		dset_names = [dset_names] * len(phase_split_path)

	dataloaders = {}
	for (phase, split_path, preprocess, mesh_path), ds_name in zip(phase_split_path, dset_names):
		img_trans = get_img_transform(args.data_stats, preprocess, args.n_channels, phase == "train")
		if ds_name == "single":
			dset = SingleImageDataset(args.base_dir, get_dir_list(split_path), args.img_size, mesh_path, args.cls, img_trans, pose_trans)
		elif ds_name == "single_tc":
			dset = SingleTCDataset(args.base_dir, get_dir_list(split_path), args.img_size, mesh_path, args.temporal_max, img_trans, pose_trans)
		elif ds_name == "cl":
			dset = CLDataset(args.base_dir, get_dir_list(split_path), args.length, args.spacing, args.img_size, mesh_path, args.skip_prev_frame, img_trans, pose_trans)
		elif ds_name == "cl_test":
			dset = TestDataset(args.base_dir, get_dir_list(split_path), args.length, args.spacing, args.img_size, mesh_path, args.skip_prev_frame, img_trans, pose_trans)
		else:
			raise NotImplementedError
		dloader = DataLoader(dset, batch_size = 1 if ds_name == "cl_test" else args.batch_size, num_workers = args.num_workers, shuffle = phase == "train")
		dataloaders[phase] = dloader

	if test:
		dataloaders = dataloaders["test"]
	loss_fun = get_loss_fun(args)
	return dataloaders, loss_fun, metric_name(loss_fun, pose_inv_trans, args.model_sel_metric, args.model_sel_rot_coef)

def get_models(args, *names):
	device = torch.device(args.device)
	res_models = []
	for t_name in names:
		if t_name.startswith("atloc"):
			if args.atloc_base == "resnet18":
				base = models.resnet18(weights = None if args.from_scratch else models.ResNet18_Weights.DEFAULT)
			elif args.atloc_base == "resnet34":
				base = models.resnet34(weights = None if args.from_scratch else models.ResNet34_Weights.DEFAULT)
			elif args.atloc_base == "resnet50":
				base = models.resnet50(weights = None if args.from_scratch else models.ResNet50_Weights.DEFAULT)
			if args.uses_posenet:
				model = PoseNet(base, output_dim = args.output_dim, droprate = args.dropout, n_channels = args.n_channels).to(device)
			else:
				model = AtLoc(base, output_dim = args.output_dim, droprate = args.dropout, scale_num_bins = args.scale_num_bins, feat_dim = args.img_encode_dim, n_channels = args.n_channels, batchnorm = args.uses_batchnorm, custombn = args.custom_bn).to(device)
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
