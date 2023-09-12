import os
import time
import torch
import argparse
from torch import nn
from copy import deepcopy
import numpy as np
import pyvista as pv
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from utils.misc import randu_gen
from models.model_utils import get_mlp
from ds_gen.depth_map_generation import get_depth_map
from utils.cl_utils import load_all_cls_npy, get_cl_dist_sum, axial_to_cl_point_ori
from utils.geometry import arbitrary_perpendicular_vector

import warnings
warnings.filterwarnings("ignore")

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--base-path", type = str, default = "./")
	parser.add_argument("--train-path", type = str, default = "train")
	parser.add_argument("--val-path", type = str, default = "val")
	parser.add_argument("-nw", "--num-workers", type = int, default = 0)
	parser.add_argument("--device", type = str, default = "cuda")
	parser.add_argument("--n-channels", type = int, default = 1)
	parser.add_argument("--num-classes", type = int, default = 3)
	parser.add_argument("--epochs", type = int, default = 30)
	parser.add_argument("--batch-size", type = int, default = 32)
	parser.add_argument("--lr", type = float, default = 1e-4)
	parser.add_argument("--save-path", type = str, default = "./checkpoints")
	parser.add_argument("--model-type", type = str, default = "resnet50")
	parser.add_argument("--dropout", type = float, default = 0.4)
	parser.add_argument("--binary", action = "store_true", default = False)
	parser.add_argument("--cap", type = float, default = 100)
	parser.add_argument("--target-size", type = int, default = 150)
	parser.add_argument("--aug", action = "store_true")
	parser.add_argument("--four-fold", action = "store_true", default = False)
	parser.add_argument("--four-thres", type = float, default = 0.4)
	parser.add_argument("--uses-bn", action = "store_true", default = False)
	parser.add_argument("--uses-sigmoid", action = "store_true", default = False)
	parser.add_argument("--reg-loss-rate", type = float, default = 0.5)
	parser.add_argument("--reg-dims", type = int, default = 1)
	parser.add_argument("--mlp-in-features", type = int, default = 2048)
	parser.add_argument("--inject-dropout", action = "store_true", default = False)
	parser.add_argument("--normalise", action = "store_true", default = False)
	parser.add_argument("--cls-neurons", type = int, nargs = "+", default = [2048, 4096, 4096])
	parser.add_argument("--reg-neurons", type = int, nargs = "+", default = [2048, 4096, 4096])
	parser.add_argument("--ckpt-path", type = str, default = "")
	parser.add_argument("--bins", type = int, default = 0)
	parser.add_argument("--rescaler-bins", type = int, default = 0)
	parser.add_argument("--dark-thres", type = float, default = 0.4)
	parser.add_argument("--sigmoid-scale", type = float, default = 12.5)
	parser.add_argument("--pool-input-size", type = int, default = 5)
	parser.add_argument("--pre-weight", type = float, default = None)
	parser.add_argument("--pool-channels", type = int, default = 2048)


	return parser.parse_args()


max_hists = [[], [], []]

def get_transform(training, args):
	blur = transforms.GaussianBlur(21, 7)
	# Original params: 50, 5, 0.1, 0.5, 220
	# Not working params: 70, 6, 0.25, 0.75, 180
	elastic = transforms.ElasticTransform(alpha = 50., sigma = 5.)
	persp = transforms.RandomPerspective(distortion_scale = 0.1, p = 0.5)
	crop_train = transforms.RandomResizedCrop(args.target_size, scale = (0.9, 1.0), ratio = (0.95, 1.05))
	resize = transforms.Resize(args.target_size)
	normalise = transforms.Normalize((0.1109, ), (0.1230, ))
	def fun(img):
		img = torch.tensor(img).unsqueeze(0)
		if training:
			img = blur(img)
			img = torch.minimum(img, torch.tensor(args.cap))
			if args.aug:
				img = elastic(img)
				img = persp(img)
				img = crop_train(img)
			else:
				img = resize(img)
		else:
			img = resize(img)
		img = (img - img.min()) / (img.max() - img.min())
		if args.bins > 0:
			img = torch.minimum(torch.floor(img * args.bins), torch.tensor(args.bins - 1)) / args.bins
		if args.normalise:
			img = normalise(img)
		return img.repeat(args.n_channels, 1, 1)
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
	def __init__(self, bins, dropout):
		super(Rescaler, self).__init__()
		self.mlp = get_mlp(bins, [32, 64, 128, 1], dropout)[0]
		self.bins = bins
	def forward(self, x):
		hst = torch.stack([torch.histc(tx.flatten(), bins = self.bins, min = 0., max = 1.) for tx in torch.unbind(x)], dim = 0)
		w = self.mlp(hst).unsqueeze(-1).unsqueeze(-1)
		out = x * w
		return out

class WeightedAvgPool(nn.Module):
	def __init__(self, args):
		super(WeightedAvgPool, self).__init__()
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		if args.pre_weight is not None:
			base = np.log(args.pre_weight - 1) if args.pre_weight > 1 else 1e-10
			self.light_weight = nn.Parameter(torch.tensor(base), requires_grad = False)
		else:
			self.light_weight = nn.Parameter(torch.tensor(0.0), requires_grad = True)
		#print("Param: ", torch.tensor(1.) + torch.exp(self.light_weight))
		self.sgm_loc, self.sgm_scale = args.dark_thres, args.sigmoid_scale
		self.pool_kernel = args.target_size // args.pool_input_size
	def forward(self, x, w):
		w = F.sigmoid((w - self.sgm_loc) * self.sgm_scale)
		w = F.avg_pool2d(w, self.pool_kernel)
		x = x.view(x.size(0), -1, w.size(2), w.size(3))
		#print("Mask avg", w.mean().item())
		w = (torch.tensor(1.) - w) + (torch.tensor(1.) + torch.exp(self.light_weight)) * w
		#print("Valued Mask avg", w.mean().item())
		#print(x.shape, w.shape)
		#print("Prev mean", x.mean().item())
		x = x * w
		#print("Weighted mean", x.mean().item())
		x = self.avgpool(x)
		return x

class ClsRegModel(nn.Module):
	def __init__(self, base, args):
		super(ClsRegModel, self).__init__()
		self.base = base
		if args.uses_bn:
			self.batch_norm = nn.BatchNorm2d(args.n_channels)
		else:
			self.batch_norm = None
		if args.rescaler_bins > 0:
			self.rescaler = Rescaler(args.rescaler_bins, args.dropout)
		else:
			self.rescaler = None

		base.fc = nn.Sequential(nn.Linear(base.fc.in_features, args.mlp_in_features), nn.Dropout(args.dropout), nn.LeakyReLU(0.2))
		self.base = base
		if args.cls_neurons != [0]:
			self.mlp_cls, cls_final_dim = get_mlp(args.mlp_in_features, args.cls_neurons, args.dropout)
		else:
			self.mlp_cls, cls_final_dim = None, args.mlp_in_features
		if args.reg_neurons != [0]:
			self.mlp_reg, reg_final_dim = get_mlp(args.mlp_in_features, args.reg_neurons, args.dropout)
		else:
			self.mlp_reg, reg_final_dim = None, args.mlp_in_features
		self.out_cls = nn.Linear(cls_final_dim, args.num_classes)
		if args.uses_sigmoid:
			self.out_reg = nn.Sequential(nn.Linear(reg_final_dim, args.reg_dims), nn.Sigmoid())
		else:
			self.out_reg = nn.Linear(reg_final_dim, args.reg_dims)
	def forward(self, x):
		if self.batch_norm is not None:
			x = self.batch_norm(x)
		if self.rescaler is not None:
			x = self.rescaler(x)
		x = self.base(x)
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
			print(model)
	else:
		model = models.swin_v2_t(weights = models.Swin_V2_T_Weights.DEFAULT, dropout = args.dropout)
		if args.n_channels != 3:
			model.features[0][0] = nn.Conv2d(args.n_channels, 96, kernel_size = 4, stride = 4)
	return ClsRegModel(model, args)

def main():
	args = get_args()
	p = pv.Plotter(off_screen = True, window_size = (224, 224))
	p.add_mesh(pv.read("./meshes/Airway_Phantom_AdjustSmooth.stl"))
	all_cls = load_all_cls_npy("./seg_cl_1")
	all_cl_sums = get_cl_dist_sum(all_cls)
	print(args)
	if args.four_fold:
		args.num_classes = 4
	val_set = PreloadDataset(os.path.join(args.base_path, f"{args.val_path}_img.npy"), os.path.join(args.base_path, f"{args.val_path}_label.npy"), get_transform(False, args), args)
	print("Val load done.", flush = True)
	val_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
	model = get_model(args)
	model = model.to(args.device)
	#model.load_state_dict(torch.load(os.path.join(args.save_path, args.ckpt_path)))
	#model.eval()
	model.base.avgpool = WeightedAvgPool()

	y_true, y_pred = [], []
	coord_trues, coord_preds = [], []
	pred_probs = []
	all_imgs = []
	for imgs, labels, coords in val_loader:
		imgs, labels, coords = imgs.to(args.device).float(), labels.to(args.device).long(), coords.to(args.device).float()
		with torch.no_grad():
			logits, pred_coords = model(imgs)
			probs = torch.softmax(logits, dim = -1)
			preds = torch.argmax(logits, dim = -1)
			y_true.append(labels.cpu().numpy())
			y_pred.append(preds.cpu().numpy())
			coord_trues.append(coords.cpu().numpy())
			coord_preds.append(pred_coords.detach().cpu().numpy())
			pred_probs.append(probs.detach().cpu().numpy())
			all_imgs.append(imgs.cpu().numpy())
	y_true, y_pred = np.concatenate(y_true, axis = 0), np.concatenate(y_pred, axis = 0)
	coord_trues, coord_preds = np.concatenate(coord_trues, axis = 0), np.concatenate(coord_preds, axis = 0)
	pred_probs = np.concatenate(pred_probs, axis = 0)
	all_imgs = np.concatenate(all_imgs, axis = 0)
	acc, f1, l1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average = "macro"), np.mean(np.abs(coord_trues - coord_preds))
	print("Res: ", acc, f1, l1)
	print("Confusion:\n", confusion_matrix(y_true, y_pred))
	scatters = [[[], [], []] for i in range(args.num_classes)]
	lb_to_colour = ["red", "green", "blue"]
	correct_l1s, wrong_l1s = [], []
	correct_yts, correct_yps, wrong_yts, wrong_yps = [], [], [], []

	"""
	for true_label, pred_label, t_c, p_c in zip(y_true, y_pred, coord_trues, coord_preds):
		scatters[true_label][0].append(t_c)
		scatters[true_label][1].append(p_c)
		scatters[true_label][2].append(lb_to_colour[pred_label])
		this_l1 = np.mean(np.abs(t_c - p_c))
		if true_label == pred_label:
			correct_l1s.append(this_l1)
			correct_yts.append(t_c)
			correct_yps.append(p_c)
		else:
			wrong_l1s.append(this_l1)
			wrong_yts.append(t_c)
			wrong_yps.append(p_c)
	"""

	window_size = 25

	prev_labels = [0] * window_size
	axial_len = 0
	prev_pred_label = 0
	momenteum = 0.8

	switch_from_axial = {0: 1, 1: 0, 2: 0}

	adjusted_pred_axials, adjusted_pred_labels, adjusted_pred_labels_prob = [], [], []

	prev_pred_probs = np.array([1, 0, 0])

	for true_label, pred_label, true_axial, pred_axial, pred_prob, input_img in zip(y_true, y_pred, coord_trues, coord_preds, pred_probs, all_imgs):
		if true_label != pred_label and true_label == 0:
			cl_pt, cl_dir = axial_to_cl_point_ori(all_cls, all_cl_sums, pred_label, pred_axial * 120)
			up = arbitrary_perpendicular_vector(cl_dir)
			rgb, dep = get_depth_map(p, cl_pt, cl_dir, up, get_outputs = True)
			dep = transforms.GaussianBlur(21, 7)(torch.tensor(dep).unsqueeze(0)).squeeze().numpy()
			plt.subplot(1, 3, 1)
			plt.imshow(dep)
			plt.subplot(1, 3, 2)
			plt.imshow(rgb)
			plt.subplot(1, 3, 3)
			plt.imshow(input_img.reshape(input_img.shape[-2], input_img.shape[-1]))
			plt.suptitle("Predicted: {} Axial: {}, Orig axial: {}".format(pred_label, pred_axial, true_axial))
			plt.show()


		prev_labels.append(pred_label)
		if len(prev_labels) > window_size:
			prev_labels.pop(0)
		values, counts = np.unique(prev_labels, return_counts = True)
		cur_label = values[np.argmax(counts)]
		strength = np.max(counts) / len(prev_labels)

		prev_pred_probs = prev_pred_probs * momenteum + pred_prob * (1 - momenteum)

		if cur_label != prev_pred_label:
			#print("Changing axial", prev_pred_label, cur_label, pred_axial)
			if strength > 0.7 or (abs(switch_from_axial[prev_pred_label] - axial_len) < 0.2):
				pass
			else:
				#print("Failed.")
				cur_label = prev_pred_label
			"""
			if prev_pred_label in [1, 2] and cur_label == 0:
				# From 1 to 0
				axial_len = 1
			else:
				# From 0 to 0
				axial_len = 0
			"""
			axial_len = pred_axial
		axial_len = momenteum * axial_len + (1 - momenteum) * pred_axial
		prev_pred_label = cur_label
		adjusted_pred_labels.append(cur_label)
		adjusted_pred_axials.append(axial_len)
		adjusted_pred_labels_prob.append(np.argmax(prev_pred_probs))

	plt.plot(y_pred, label = "Pred")
	plt.plot(y_true, label = "True")
	plt.title("Types")
	plt.show()
	
	plt.plot(coords_pred, label = "Pred")
	plt.plot(coords_true, label = "True")
	plt.title("Coords")
	plt.show()


	print("Adjusted: acc {:.4f} f1 {:.4f} l1 {:.4f}".format(accuracy_score(y_true, adjusted_pred_labels), f1_score(y_true, adjusted_pred_labels, average = "macro"), np.mean(np.abs(adjusted_pred_axials - coord_trues))))
	print("Adjusted with prob acc: {:.4f} f1: {:.4f}.".format(accuracy_score(y_true, adjusted_pred_labels_prob), f1_score(y_true, adjusted_pred_labels_prob, average = "macro")))


	print(np.corrcoef(coord_trues.ravel(), coord_preds.ravel())[1][0], np.corrcoef(np.array(correct_yts).ravel(), np.array(correct_yps).ravel())[1][0], np.corrcoef(np.array(wrong_yts).ravel(), np.array(wrong_yps).ravel())[1][0])

	print("Correct L1s: {:.4f}, Wrong L1s: {:.4f}".format(np.mean(correct_l1s), np.mean(wrong_l1s)))

	"""
	for i in range(args.num_classes):
		plt.scatter(scatters[i][0], scatters[i][1], color = scatters[i][2])
		plt.title(f"Class: {i}")
		plt.show()
	"""
if __name__ == '__main__':
	main()
