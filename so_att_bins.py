import os
import time
import torch
import argparse
from torch import nn
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from utils.misc import randu_gen
from models.model_utils import get_mlp
from torchvision.models.resnet import Bottleneck
from models.atloc import AtLoc

import warnings
warnings.filterwarnings("ignore")

hidden_args = ["base_path", "val_path", "num_workers", "device", "n_channels", "epochs", "save_path", "binary", "four_fold", "four_thres",
	"uses_sigmoid", "reg_dims", "mlp_in_features", "inject_dropout", "cls_neurons", "reg_neurons", "amsgrad", "normalise", "bins", "dark_thres",
	"sigmoid_scale", "pool_input_size", "pool_channels", "num_classes", "resume", "border_padding", "relative_values", "stretch_loss_rate"]

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument("--base-path", type = str, default = "./")
	parser.add_argument("--train-path", type = str, default = "train")
	parser.add_argument("--val-path", type = str, default = "val")
	parser.add_argument("-nw", "--num-workers", type = int, default = 0)
	parser.add_argument("--device", type = str, default = "cuda")
	parser.add_argument("--n-channels", type = int, default = 1)
	parser.add_argument("--num-classes", type = int, default = 3)
	parser.add_argument("--epochs", type = int, default = 20)
	parser.add_argument("--batch-size", type = int, default = 32)
	parser.add_argument("--lr", type = float, default = 1e-4)
	parser.add_argument("--save-path", type = str, default = "./checkpoints")
	parser.add_argument("--model-type", type = str, default = "resnet101")
	parser.add_argument("--dropout", type = float, default = 0.2)
	parser.add_argument("--binary", action = "store_true", default = False)
	parser.add_argument("--cap", type = float, default = 100)
	parser.add_argument("--target-size", type = int, default = 150)
	parser.add_argument("--aug", action = "store_true", default = False)
	parser.add_argument("--aug-old", action = "store_true", default = False)
	parser.add_argument("--four-fold", action = "store_true", default = False)
	parser.add_argument("--four-thres", type = float, default = 0.4)
	parser.add_argument("--uses-bn", action = "store_true", default = False)
	parser.add_argument("--uses-sigmoid", action = "store_true", default = False)
	parser.add_argument("--reg-loss-rate", type = float, default = 0.5)
	parser.add_argument("--reg-dims", type = int, default = 1)
	parser.add_argument("--mlp-in-features", type = int, default = 2048)
	parser.add_argument("--inject-dropout", action = "store_true", default = False)
	parser.add_argument("--normalise", action = "store_true", default = False)
	parser.add_argument("--cls-neurons", type = int, nargs = "+", default = [2048, 2048, 2048])
	parser.add_argument("--reg-neurons", type = int, nargs = "+", default = [2048, 2048, 2048])
	parser.add_argument("--amsgrad", action = "store_true", default = False)
	parser.add_argument("--step-lr-size", type = int, default = 200)
	parser.add_argument("--bins", type = int, default = 0)
	parser.add_argument("--rescaler-bins", type = int, default = 20)
	parser.add_argument("--dark-thres", type = float, default = 0.4)
	parser.add_argument("--sigmoid-scale", type = float, default = 12.5)
	parser.add_argument("--pool-input-size", type = int, default = 5)
	parser.add_argument("--pre-weight", type = float, default = 5)
	parser.add_argument("--pool-channels", type = int, default = 2048)
	parser.add_argument("--resume", type = str, default = None)
	parser.add_argument("--adv-scale", type = float, default = 0)
	# Normal adv scale 0.02
	parser.add_argument("--dark-hist-rate", type = float, default = 0.2)
	parser.add_argument("--border-padding", type = int, default = 0)
	parser.add_argument("--relative-values", action = "store_true")
	parser.add_argument("--stretch-loss-rate", type = float, default = 0.)
	parser.add_argument("--pool-dropout", type = float, default = 0.1)
	parser.add_argument("--mlp-dropout", type = float, default = 0.1)
	parser.add_argument("--dont-rescale", action = "store_true")
	parser.add_argument("--dont-weighted-mask", action = "store_true")
	parser.add_argument("--da-type", type = str, default = "min-max")
	parser.add_argument("--uses-atloc", action = "store_true")
	parser.add_argument("--val-blur", action = "store_true")
	return parser

def get_args():
	parser = get_parser()
	return parser.parse_args()


def get_transform(training, args):
	blur = transforms.GaussianBlur(21, 7)
	# Original params: 50, 5, 0.1, 0.5, 220
	# Not working params: 70, 6, 0.25, 0.75, 180
	elastic = transforms.ElasticTransform(alpha = 50., sigma = 5.)
	persp = transforms.RandomPerspective(distortion_scale = 0.15, p = 0.5)
	crop_train = transforms.RandomResizedCrop(args.target_size, scale = (0.9, 1.0), ratio = (0.95, 1.05))
	resize = transforms.Resize(args.target_size)
	mask_resize = transforms.Resize(args.pool_input_size)
	normalise = transforms.Normalize((0.1109, ), (0.1230, ))
	jitter = transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0, hue = 0)
	def fun(img):
		input_img = img
		img = torch.tensor(img).unsqueeze(0)
		if training:
			img = torch.minimum(img, torch.tensor(args.cap))
			if args.aug:
				img = blur(img)
				img = elastic(img)
				img = persp(img)
				img = crop_train(img)
			else:
				if args.val_blur:
					img = blur(img)
				img = resize(img)
		else:
			img = resize(img)
		stretch = img.max() - img.min()
		#print(stretch)
		if args.da_type == "min-max":
			img = (img - img.min()) / stretch
		elif args.da_type == "linear":
			if training:
				_w, _b = 0.13236457109451294, 2.5300467014312744
				img = img * _w + _b
		elif args.da_type == "quantile":
			orig_shape = img.shape
			q = torch.argsort(img.flatten())
			q = torch.argsort(q) / torch.max(q)
			img = q.view(orig_shape)
		elif args.da_type == "hist_simple":
			num_bins = 20
			img = (img - img.min()) / stretch
			img = torch.minimum(torch.floor(img * num_bins), torch.tensor(num_bins - 1)) / num_bins
		elif args.da_type == "hist_complex":
			num_bins = 20
			orig_shape = img.shape
			img = (img - img.min()) / stretch
			img = img.numpy().squeeze()
			img_hist_indices = np.minimum(np.floor(img * num_bins).astype(int), num_bins - 1)
			img_hist_heights = np.histogram(img.ravel(), bins = num_bins, density = True)[0]
			hist_peak_idx = np.argmax(img_hist_heights)
			img_hist_heights = img_hist_heights / img_hist_heights[hist_peak_idx]
			for j in range(len(img_hist_heights)):
				i = len(img_hist_heights) - j - 1
				if i < hist_peak_idx:
					img_hist_heights[i] = 1
				if j > 0:
					img_hist_heights[i] = max(img_hist_heights[i], img_hist_heights[i + 1])
			labels = 1.0 - img_hist_heights[img_hist_indices]
			img = torch.tensor(labels.reshape(orig_shape))
		else:
			raise NotImplementedError

		if args.adv_scale > 0:
			#plt.subplot(1, 3, 1)
			#plt.imshow(img.numpy().squeeze())
			"""
			t_hst = torch.histc(img.flatten(), bins = args.rescaler_bins, min = 0., max = 1.)
			comp_height = torch.max(t_hst) * args.dark_hist_rate
			comp_height_bin = torch.argmax(t_hst)
			t_value = torch.argmin(torch.logical_or((torch.arange(args.rescaler_bins) < comp_height_bin), (t_hst > comp_height)).float()) / args.rescaler_bins
			w = F.sigmoid((img - t_value) * args.sigmoid_scale)
			noise = torch.randn_like(img) * (1 - w) * args.adv_scale
			"""
			noise = torch.randn_like(img) * (1 - img) * args.adv_scale
			img = img + noise
			"""
			plt.subplot(1, 3, 2)
			plt.imshow(img.numpy().squeeze())
			plt.subplot(1, 3, 3)
			plt.imshow(noise.numpy().squeeze())
			plt.show()
			"""
		if args.aug_old:
			img = jitter(img)

		img = transforms.functional.pad(img, args.border_padding)

		"""
		plt.subplot(1, 2, 1)
		plt.imshow(input_img)
		plt.subplot(1, 2, 2)
		plt.imshow(img.numpy().squeeze())
		plt.show()
		"""

		return img
	return fun

class PreloadDataset(Dataset):
	def __init__(self, img_path, label_path, transform, args):
		super(PreloadDataset, self).__init__()
		self.img_data = np.load(img_path)
		self.label_data = np.load(label_path)
		self.transform = transform
		self.four_fold = args.four_fold
		self.four_thres = args.four_thres
		self.reg_dims = args.reg_dims
		"""
		for i in range(len(self.img_data)):
			max_hists[self.label_data[i, ...]].append(self.img_data[i, ...].max())
		"""

	def __getitem__(self, idx):
		lb = int(self.label_data[idx, 0])
		reg = self.label_data[idx, 1 : 1 + self.reg_dims]
		img = self.transform(self.img_data[idx, ...])
		if self.four_fold:
			if lb == 0 and self.label_data[idx, 1] > self.four_thres:
				lb = 3

		"""
		if lb == 0:
			plt.imshow(img.numpy()[0, ...])
			plt.title(lb, self.label_data[idx, 1])
			plt.show()
		"""

		return img, lb, reg
	def __len__(self):
		return len(self.img_data)

class Rescaler(nn.Module):
	def __init__(self, bins, dropout, height_rate, dont_rescale, v_range):
		super(Rescaler, self).__init__()
		self.mlp = get_mlp(bins, [32, 64, 128, 1], dropout)[0]
		self.bins = bins
		self.height_rate = height_rate
		self.dont_rescale = dont_rescale
		self.v_range = v_range
	def forward(self, x):
		hst = []
		value = []
		for tx in torch.unbind(x):
			t_hst = torch.histc(tx.flatten(), bins = self.bins, min = 0., max = self.v_range)
			comp_height = torch.max(t_hst) * self.height_rate
			comp_height_bin = torch.argmax(t_hst)
			t_value = torch.argmin(torch.logical_or((torch.arange(self.bins).to(t_hst.device) < comp_height_bin), (t_hst > comp_height)).float()) / self.bins * self.v_range
			#print(t_hst, t_value)
			hst.append(t_hst)
			value.append(t_value)
		hst = torch.stack(hst, dim = 0).detach()
		value = torch.stack(value, dim = 0).detach()
		w = self.mlp(hst).unsqueeze(-1).unsqueeze(-1)
		if not self.dont_rescale:
			out = x * w
		else:
			out = x
		return out, value

class WeightedAvgPool(nn.Module):
	def __init__(self, args):
		super(WeightedAvgPool, self).__init__()
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.relative_values = args.relative_values
		self.dropout = nn.Dropout2d(args.pool_dropout)
	def forward(self, x, w):
		#print("Valued Mask avg", w.mean().item())
		#print("Prev mean", x.mean().item())
		#x = self.dropout(x * w)
		#print(x.min(), w.min())
		#print("W range: ", w.min(), w.max(), w.mean())
		#assert x.min() >= 0
		#assert w.min() >= 0


		x = x.view(x.size(0), -1, w.size(2), w.size(3))
		#w = w / torch.sum(w, dim = (1, 2, 3), keepdim = True)
		#print(x.shape, w.shape)
		x = self.dropout(x * w)
		#print("Weighted mean", x.mean().item())
		x = torch.sum(x, dim = (2, 3), keepdim = True)
		#print(x.shape)
		return x

class ClsRegModel(nn.Module):
	def __init__(self, base, mask_extractor, args):
		super(ClsRegModel, self).__init__()
		if args.pre_weight is not None:
			bb = np.log(args.pre_weight - 1) if args.pre_weight > 1 else 1e-10
			self.light_weight = nn.Parameter(torch.tensor(bb), requires_grad = False)
		else:
			self.light_weight = nn.Parameter(torch.tensor(0.0), requires_grad = True)
		#print("Param: ", torch.tensor(1.) + torch.exp(self.light_weight))
		self.sgm_loc, self.sgm_scale = args.dark_thres, args.sigmoid_scale
		self.pool_kernel = args.target_size // args.pool_input_size
		self.adv_scale = args.adv_scale
		self.pool_input_size = args.pool_input_size
		self.target_size = args.target_size
		self.border_padding = args.border_padding
		self.dont_rescale = args.dont_rescale
		self.dont_weighted_mask = args.dont_weighted_mask
		self.n_channels = args.n_channels

		self.dark_thres = args.dark_thres
		self.base = base
		if args.uses_bn:
			self.batch_norm = nn.BatchNorm2d(args.n_channels)
		else:
			self.batch_norm = None
		self.v_range = 100.0 if args.da_type == "linear" else 1.0
		if args.rescaler_bins > 0:
			print("Using rescaler.")
			self.rescaler = Rescaler(args.rescaler_bins, args.mlp_dropout, args.dark_hist_rate, args.dont_rescale, self.v_range)
		else:
			self.rescaler = None

		self.base_pool = WeightedAvgPool(args)
		self.base_fc = nn.Sequential(nn.Linear(base.fc.in_features, args.mlp_in_features), nn.Dropout(args.dropout), nn.LeakyReLU(0.2))
		base.avgpool = nn.Identity()
		base.fc = nn.Identity()
		self.base = base
		mask_extractor.avgpool = nn.Identity()
		mask_extractor.fc = nn.Identity()
		self.mask_extractor = mask_extractor
		if args.cls_neurons != [0]:
			self.mlp_cls, cls_final_dim = get_mlp(args.mlp_in_features, args.cls_neurons, args.mlp_dropout)
		else:
			self.mlp_cls, cls_final_dim = None, args.mlp_in_features
		if args.reg_neurons != [0]:
			self.mlp_reg, reg_final_dim = get_mlp(args.mlp_in_features, args.reg_neurons, args.mlp_dropout)
		else:
			self.mlp_reg, reg_final_dim = None, args.mlp_in_features
		self.out_cls = nn.Linear(cls_final_dim, args.num_classes)
		if args.uses_sigmoid:
			self.out_reg = nn.Sequential(nn.Linear(reg_final_dim, args.reg_dims), nn.Sigmoid())
		else:
			self.out_reg = nn.Linear(reg_final_dim, args.reg_dims)
	def forward(self, x):
		x1 = x
		if self.rescaler is not None:
			x, hv = self.rescaler(x)
			hv = hv.view(-1, 1, 1, 1)

		if not self.dont_weighted_mask:
			w_lg = F.sigmoid(((x1 - hv) / self.v_range) * self.sgm_scale)
			totally_white = torch.ones_like(x1)
			with torch.no_grad():
				w_sm = self.mask_extractor(w_lg)
				totally_white = self.mask_extractor(totally_white)
			w_sm = w_sm.view(w_sm.size(0), 1, self.pool_input_size, self.pool_input_size)
			totally_white = totally_white.view(totally_white.size(0), 1, self.pool_input_size, self.pool_input_size)
			w_sm = w_sm / totally_white
			mins, maxes = w_sm.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0], w_sm.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0]
			w_sm = (w_sm - mins) / (maxes - mins)
			w_sm = (torch.tensor(1.) - w_sm) + (torch.tensor(1.) + torch.exp(self.light_weight)) * w_sm
		else:
			w_sm = torch.ones(x.size(0), 1, self.pool_input_size, self.pool_input_size).to(x.device)

		w_sm = w_sm / torch.sum(w_sm, dim = (1, 2, 3), keepdim = True)

		if self.batch_norm is not None:
			x = self.batch_norm(x)
		x = self.base(x.repeat(1, self.n_channels, 1, 1))
		#print("Forward: ", x.shape)
		x = self.base_pool(x, w_sm)
		#print(x.shape)
		x = x.view(x.size(0), -1)
		x = self.base_fc(x)
		x_cls = self.mlp_cls(x) if self.mlp_cls is not None else x
		x_reg = self.mlp_reg(x) if self.mlp_reg is not None else x
		out_cls = self.out_cls(x_cls)
		out_reg = self.out_reg(x_reg)
		return out_cls, out_reg

def append_dropout(model, dropout):
	for name, module in model.named_children():
		if len(list(module.children())) > 0:
			append_dropout(module, dropout)
		if isinstance(module, nn.ReLU):
			new_module = nn.Sequential(module, nn.Dropout2d(dropout))
			setattr(model, name, new_module)

class HalfingLayer(nn.Module):
	def __init__(self):
		super(HalfingLayer, self).__init__()
	def forward(self, x):
		return x / 2

def change_n_channels(model):
	for name, module in model.named_children():
		if len(list(module.children())) > 0:
			change_n_channels(module)
		if isinstance(module, nn.Conv2d):
			#print("Changed to new model")
			kernel_size = module.kernel_size
			sum_field = kernel_size[0] * kernel_size[1]
			new_module = nn.Conv2d(1, 1, kernel_size = kernel_size, stride = module.stride, padding = module.padding, groups = module.groups, bias = False, dilation = module.dilation)
			new_module.weight = nn.Parameter(torch.ones_like(new_module.weight) / sum_field, requires_grad = False)
			#print(new_module.weight)
			#print(new_module.padding)
			setattr(model, name, new_module)
		elif isinstance(module, nn.BatchNorm2d):
			new_module = nn.Identity()
			setattr(model, name, new_module)
		elif isinstance(module, Bottleneck):
			new_module = nn.Sequential(module, HalfingLayer())
			setattr(model, name, new_module)

def net_to_receptive_extractor(model):
	model = deepcopy(model)
	change_n_channels(model)
	return model

def get_model(args):
	if args.model_type.find("resnet") != -1:
		if args.model_type == "resnet34":
			model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
		elif args.model_type == "resnet18":
			model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
		elif args.model_type == "resnet50":
			model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
		elif args.model_type == "resnet101":
			model = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
		elif args.model_type == "resnet152":
			model = models.resnet152(weights = models.ResNet152_Weights.DEFAULT)
		else:
			raise NotImplementedError

		if args.n_channels != 3:
			model.conv1 = nn.Conv2d(args.n_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
		else:
			print("Not replacing the first layer.")
		if args.inject_dropout:
			append_dropout(model, args.dropout)
			#print(model)
	else:
		model = models.swin_v2_t(weights = models.Swin_V2_T_Weights.DEFAULT, dropout = args.dropout)
		if args.n_channels != 3:
			model.features[0][0] = nn.Conv2d(args.n_channels, 96, kernel_size = 4, stride = 4)
	mask_extractor = net_to_receptive_extractor(model)
	#print(mask_extractor)

	if not args.uses_atloc:
		return ClsRegModel(model, mask_extractor, args)
	else:
		return AtLoc(model, output_dim = args.num_classes, reg_dim = args.reg_dims, droprate=0.5, scale_num_bins = 0, batchnorm = False, custombn = False, pretrained=True, feat_dim=2048, n_channels = 1, lstm=False)


def main():
	args = get_args()
	for arg_name, arg_value in vars(args).items():
		if arg_name not in hidden_args:
			print(f"{arg_name}: {arg_value}")

	if args.four_fold:
		args.num_classes = 4
	train_set = PreloadDataset(os.path.join(args.base_path, f"{args.train_path}_img.npy"), os.path.join(args.base_path, f"{args.train_path}_label.npy"), get_transform(True, args), args)
	print("Train load done.", flush = True)
	val_set = PreloadDataset(os.path.join(args.base_path, f"{args.val_path}_img.npy"), os.path.join(args.base_path, f"{args.val_path}_label.npy"), get_transform(False, args), args)
	print("Val load done.", flush = True)
	train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
	val_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
	model = get_model(args)
	model = model.to(args.device)

	if args.resume is not None:
		print("Resuming from ", args.resume, flush = True)
		model.load_state_dict(torch.load(os.path.join(args.save_path, args.resume)))
	optimiser = torch.optim.Adam(model.parameters(), lr = args.lr, amsgrad = args.amsgrad)
	scheduler = torch.optim.lr_scheduler.StepLR(optimiser, args.step_lr_size)
	max_acc = 0
	for epoch in range(args.epochs):
		st_time = time.time()
		for phase, loader in [("train", train_loader), ("val", val_loader)]:
			if phase == "train":
				model.train()
			else:
				model.eval()
			y_true, y_pred = [], []
			coord_trues, coord_preds = [], []
			sum_loss = num_loss = 0
			for imgs, labels, coords in loader:
				#print("Batch")
				imgs, labels, coords = imgs.to(args.device).float(), labels.to(args.device).long(), coords.to(args.device).float()
				with torch.set_grad_enabled(phase == "train"):
					logits, pred_coords = model(imgs)
					loss = nn.CrossEntropyLoss()(logits, labels) + args.reg_loss_rate * nn.MSELoss()(coords, pred_coords)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				sum_loss += loss.item() * imgs.size(0)
				num_loss += imgs.size(0)
				preds = torch.argmax(logits, dim = -1)
				y_true.append(labels.cpu().numpy())
				y_pred.append(preds.cpu().numpy())
				coord_trues.append(coords.cpu().numpy())
				coord_preds.append(pred_coords.detach().cpu().numpy())
			y_true, y_pred = np.concatenate(y_true, axis = 0), np.concatenate(y_pred, axis = 0)
			coord_trues, coord_preds = np.concatenate(coord_trues, axis = 0), np.concatenate(coord_preds, axis = 0)
			loss, acc, f1, l1 = sum_loss / num_loss, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average = "macro"), np.mean(np.abs(coord_trues - coord_preds))
			print("Epoch {} phase {}: loss = {:.4f}, accuracy = {:.4f}, f1 = {:.4f}, reg L1 = {:.4f}".format(epoch, phase, loss, acc, f1, l1), flush = True)
			if phase == "val":
				if acc > max_acc:
					max_acc, max_f1, max_l1, best_weights = acc, f1, l1, deepcopy(model.state_dict())
				torch.save(model.state_dict(), os.path.join(args.save_path, f"ckpt-{acc:.4f}-{l1:.4f}-{epoch}.pt"))
		print("Epoch time: {:.2f} mins".format((time.time() - st_time) / 60))
		scheduler.step()
	print("Max: ", max_acc, max_f1, max_l1)

if __name__ == '__main__':
	main()
