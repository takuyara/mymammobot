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

import warnings
warnings.filterwarnings("ignore")

hidden_args = ["base_path", "val_path", "num_workers", "device", "n_channels", "epochs", "save_path", "binary", "cap", "target_size", "four_fold", "four_thres",
	"uses_sigmoid", "reg_dims", "mlp_in_features", "inject_dropout", "normalise", "bins", "dark_thres", "sigmoid_scale", "pool_input_size", "pool_channels", "num_classes", "aug"]

def get_args():
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
	parser.add_argument("--dark-hist-rate", type = float, default = 0.2)
	parser.add_argument("--border-padding", type = int, default = 0)
	parser.add_argument("--relative-values", action = "store_true")
	parser.add_argument("--stretch-loss-rate", type = float, default = 0.)
	parser.add_argument("--pool-dropout", type = float, default = 0.1)
	parser.add_argument("--mlp-dropout", type = float, default = 0.1)
	parser.add_argument("--dont-rescale", action = "store_true")
	parser.add_argument("--dont-weighted-mask", action = "store_true")
	return parser.parse_args()


max_hists = [[], [], []]

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
	def fun(img):
		img = torch.tensor(img).unsqueeze(0)
		if training:
			if args.aug:
				img = blur(img)
				img = torch.minimum(img, torch.tensor(args.cap))
				img = elastic(img)
				img = persp(img)
				img = crop_train(img)
			else:
				img = resize(img)
		else:
			img = resize(img)
		stretch = img.max() - img.min()
		#print(stretch)
		img = (img - img.min()) / stretch
		img = transforms.functional.pad(img, args.border_padding)
		if args.bins > 0:
			print("Using histogram to reduce noise.")
			img = torch.minimum(torch.floor(img * args.bins), torch.tensor(args.bins - 1)) / args.bins
		return img.repeat(args.n_channels, 1, 1), stretch
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

		return *img, lb, reg
	def __len__(self):
		return len(self.img_data)

class Rescaler(nn.Module):
	def __init__(self, bins, dropout, height_rate, dont_rescale):
		super(Rescaler, self).__init__()
		self.mlp = get_mlp(bins, [32, 64, 128, 1], dropout)[0]
		self.bins = bins
		self.height_rate = height_rate
		self.dont_rescale = dont_rescale
	def forward(self, x):
		hst = []
		value = []
		for tx in torch.unbind(x):
			t_hst = torch.histc(tx.flatten(), bins = self.bins, min = 0., max = 1.)
			comp_height = torch.max(t_hst) * self.height_rate
			comp_height_bin = torch.argmax(t_hst)
			t_value = torch.argmin(torch.logical_or((torch.arange(self.bins).to(t_hst.device) < comp_height_bin), (t_hst > comp_height)).float()) / self.bins
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
		return out, w, value

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

class PlaceHolder(nn.Module):
	def __init__(self):
		super(PlaceHolder, self).__init__()
	def forward(self, x):
		print("Input Placeholder: ", x.shape)
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

		self.dark_thres = args.dark_thres
		self.base = base
		if args.uses_bn:
			self.batch_norm = nn.BatchNorm2d(args.n_channels)
		else:
			self.batch_norm = None
		if args.rescaler_bins > 0:
			print("Using rescaler.")
			self.rescaler = Rescaler(args.rescaler_bins, args.mlp_dropout, args.dark_hist_rate, args.dont_rescale)
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
			x, stretch, hv = self.rescaler(x)
			hv = hv.view(-1, 1, 1, 1)
		else:
			stretch = torch.zeros(x.size(0), 1).to(x.device)

		if not self.dont_weighted_mask:
			w_lg = F.sigmoid((x1 - hv) * self.sgm_scale)
			totally_white = torch.ones_like(x1)
			with torch.no_grad():
				w_sm = self.mask_extractor(w_lg)
				totally_white = self.mask_extractor(totally_white)
			w_sm = w_sm.view(w_sm.size(0), 1, self.pool_input_size, self.pool_input_size)
			totally_white = totally_white.view(totally_white.size(0), 1, self.pool_input_size, self.pool_input_size)
			w_sm = w_sm / totally_white
			rela_weights = w_sm
			mins, maxes = w_sm.min(-1, keepdim = True)[0].min(-2, keepdim = True)[0], w_sm.max(-1, keepdim = True)[0].max(-2, keepdim = True)[0]
			w_sm = (w_sm - mins) / (maxes - mins)
			w_sm_before = w_sm
			w_sm = (torch.tensor(1.) - w_sm) + (torch.tensor(1.) + torch.exp(self.light_weight)) * w_sm

			
			light_weight_1 = torch.tensor(0.)
			w_sm_2 = (torch.tensor(1.) - w_sm_before) + (torch.tensor(1.) + torch.exp(light_weight_1)) * w_sm_before
			w_sm = w_sm / torch.sum(w_sm, dim = (1, 2, 3), keepdim = True)
			w_sm_2 = w_sm_2 / torch.sum(w_sm_2, dim = (1, 2, 3), keepdim = True)
			w_sm_0 = w_sm_before / torch.sum(w_sm_before, dim = (1, 2, 3), keepdim = True)
			for img, t_msk, t_white, t_rela, t_final_0, t_final_2, t_final_7 in zip(torch.unbind(x1), torch.unbind(w_lg), torch.unbind(totally_white), torch.unbind(rela_weights), torch.unbind(w_sm_0), torch.unbind(w_sm_2), torch.unbind(w_sm)):
				fig, axes = plt.subplots(2, 3)
				axes = axes.ravel()
				for i, dt in enumerate([img, t_msk, t_white, t_rela, t_final_2, t_final_7]):
					dt = dt.cpu().numpy().squeeze()
					axes[i].imshow(dt)
					axes[i].axis("off")
					if i > 1:
						for x in range(5):
							for y in range(5):
								if i > 3:
									c = "{:.3f}".format(dt[y, x] * 100)
								else:
									c = "{:.3f}".format(dt[y, x])
								axes[i].text(x, y, c, va = "center", ha = "center", color = "red", fontsize = 15)
				plt.show()

		else:
			w_sm = torch.ones(x.size(0), 1, self.pool_input_size, self.pool_input_size).to(x.device)

		w_sm = w_sm / torch.sum(w_sm, dim = (1, 2, 3), keepdim = True)

		if self.batch_norm is not None:
			x = self.batch_norm(x)
		x = self.base(x)
		#print("Forward: ", x.shape)
		x = self.base_pool(x, w_sm)
		#print(x.shape)
		x = x.view(x.size(0), -1)
		x = self.base_fc(x)
		x_cls = self.mlp_cls(x) if self.mlp_cls is not None else x
		x_reg = self.mlp_reg(x) if self.mlp_reg is not None else x
		out_cls = self.out_cls(x_cls)
		out_reg = self.out_reg(x_reg)
		return out_cls, out_reg, stretch

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
		if args.inject_dropout:
			append_dropout(model, args.dropout)
			#print(model)
	else:
		model = models.swin_v2_t(weights = models.Swin_V2_T_Weights.DEFAULT, dropout = args.dropout)
		if args.n_channels != 3:
			model.features[0][0] = nn.Conv2d(args.n_channels, 96, kernel_size = 4, stride = 4)
	mask_extractor = net_to_receptive_extractor(model)
	#print(mask_extractor)

	return ClsRegModel(model, mask_extractor, args)

def main():
	args = get_args()
	print(args)
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
			stretch_trues, stretch_preds = [], []
			sum_loss = num_loss = 0
			for imgs, stretches, labels, coords in loader:
				#print("Batch")
				imgs, stretches, labels, coords = imgs.to(args.device).float(), stretches.to(args.device).float(), labels.to(args.device).long(), coords.to(args.device).float()
				with torch.set_grad_enabled(phase == "train"):
					logits, pred_coords, pred_stretches = model(imgs)
					loss = nn.CrossEntropyLoss()(logits, labels) + args.reg_loss_rate * nn.MSELoss()(coords, pred_coords) + args.stretch_loss_rate * nn.MSELoss()(stretches, pred_stretches)
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
				stretch_trues.append(stretches.cpu().numpy())
				stretch_preds.append(pred_stretches.detach().cpu().numpy())
			y_true, y_pred = np.concatenate(y_true, axis = 0), np.concatenate(y_pred, axis = 0)
			coord_trues, coord_preds = np.concatenate(coord_trues, axis = 0), np.concatenate(coord_preds, axis = 0)
			stretch_trues, stretch_preds = np.concatenate(stretch_trues, axis = 0), np.concatenate(stretch_preds, axis = 0)
			loss, acc, f1, l1 = sum_loss / num_loss, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average = "macro"), np.mean(np.abs(coord_trues - coord_preds))
			stre_l1 = np.mean(np.abs(stretch_trues - stretch_preds))
			print("Epoch {} phase {}: loss = {:.4f}, accuracy = {:.4f}, f1 = {:.4f}, reg L1 = {:.4f}, stre l1 = {:.2f}".format(epoch, phase, loss, acc, f1, l1, stre_l1), flush = True)
			if phase == "val":
				if acc > max_acc:
					max_acc, max_f1, max_l1, best_weights = acc, f1, l1, deepcopy(model.state_dict())
				torch.save(model.state_dict(), os.path.join(args.save_path, f"ckpt-{acc:.4f}-{l1:.4f}-{epoch}.pt"))
		print("Epoch time: {:.2f} mins".format((time.time() - st_time) / 60))
		scheduler.step()
	print("Max: ", max_acc, max_f1, max_l1)

if __name__ == '__main__':
	main()
