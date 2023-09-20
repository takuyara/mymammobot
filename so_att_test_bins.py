import os
import time
import torch
import argparse
from torch import nn
from copy import deepcopy
import numpy as np
import pyvista as pv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from utils.misc import randu_gen
from models.model_utils import get_mlp
from ds_gen.depth_map_generation import get_depth_map
from utils.cl_utils import load_all_cls_npy, get_cl_dist_sum, axial_to_cl_point_ori
from utils.geometry import arbitrary_perpendicular_vector
from so_att_bins import get_model, get_parser, PreloadDataset, get_transform

import warnings
warnings.filterwarnings("ignore")

def get_args():
	parser = get_parser()
	parser.add_argument("--ckpt-path", type = str, default = "")
	return parser.parse_args()

def get_yt_yp_list(args, val_path, model):
	val_set = PreloadDataset(os.path.join(args.base_path, f"{val_path}_img.npy"), os.path.join(args.base_path, f"{val_path}_label.npy"), get_transform(False, args), args)
	val_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
	y_true, y_pred = [], []
	coord_trues, coord_preds = [], []
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
	y_true, y_pred = np.concatenate(y_true, axis = 0), np.concatenate(y_pred, axis = 0)
	coord_trues, coord_preds = np.concatenate(coord_trues, axis = 0), np.concatenate(coord_preds, axis = 0)
	return y_true, y_pred, coord_trues, coord_preds

def smoothing(y_pred, coord_preds, window_size = 25, momenteum = 0.8):
	prev_labels = [0] * window_size
	axial_len = 0
	prev_pred_label = 0

	switch_from_axial = {0: 1, 1: 0, 2: 0}

	adjusted_pred_axials, adjusted_pred_labels = [], []

	for pred_label, pred_axial in zip(y_pred, coord_preds):
		prev_labels.append(pred_label)
		if len(prev_labels) > window_size:
			prev_labels.pop(0)
		values, counts = np.unique(prev_labels, return_counts = True)
		cur_label = values[np.argmax(counts)]
		strength = np.max(counts) / len(prev_labels)

		if cur_label != prev_pred_label:
			if strength > 0.7 or (abs(switch_from_axial[prev_pred_label] - axial_len) < 0.2):
				pass
			else:
				cur_label = prev_pred_label
			axial_len = pred_axial
		
		axial_len = momenteum * axial_len + (1 - momenteum) * pred_axial
		prev_pred_label = cur_label
		adjusted_pred_labels.append(cur_label)
		adjusted_pred_axials.append(axial_len)
	return adjusted_pred_labels, adjusted_pred_axials

def get_metrics(y_true, y_pred, coord_trues, coord_preds):
	acc, f1, l1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average = "macro"), np.mean(np.abs(coord_trues - coord_preds))
	conf = confusion_matrix(y_true, y_pred)
	return {"acc": acc, "f1": f1, "l1": l1, "confusion": conf}

def get_serial(val_path, model, args, smooth = False):
	y_true, y_pred, coord_trues, coord_preds = get_yt_yp_list(args, val_path, model)
	if smooth:
		a_labels, a_axials = smoothing(y_pred, coord_preds)
		return get_metrics(y_true, y_pred, coord_trues, coord_preds), get_metrics(y_true, a_labels, coord_trues, a_axials)
	else:
		return get_metrics(y_true, y_pred, coord_trues, coord_preds)

def outit(dct):
	return "Acc: {:.2f}, F1: {:.2f}, L1: {:.2f}".format(dct["acc"] * 100, dct["f1"] * 100, dct["l1"] * 100)

def merge_multi(outs):
	return {k : (sum([o[k] for o in outs]) / (3 if k != "confusion" else 1)) for k in ["acc", "f1", "l1", "confusion"]}

def main():
	args = get_args()
	"""
	p = pv.Plotter(off_screen = True, window_size = (224, 224))
	p.add_mesh(pv.read("./meshes/Airway_Phantom_AdjustSmooth.stl"))
	all_cls = load_all_cls_npy("./seg_cl_1")
	all_cl_sums = get_cl_dist_sum(all_cls)
	"""
	print(args.ckpt_path)
	if args.four_fold:
		args.num_classes = 4

	model = get_model(args)
	model = model.to(args.device)
	model.load_state_dict(torch.load(os.path.join(args.save_path, args.ckpt_path)))
	model.eval()

	print("Val confirmed:")
	print(outit(get_serial("val", model, args)))

	print("Val all:")
	print(outit(get_serial("val_all", model, args)))

	ds = []
	#print("Val REAL-0")
	ds.append(get_serial("val_all_0", model, args, smooth = True)[1])

	#print("Val REAL-1")
	ds.append(get_serial("val_all_1", model, args, smooth = True)[1])

	#print("Val REAL-2")
	ds.append(get_serial("val_all_2", model, args, smooth = True)[1])

	print("Smoothed:")
	print(outit(merge_multi(ds)))


def depr():

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

	#plt.figure(figsize = (9, 3.5))
	#plt.subplots_adjust(left = 0.01, right = 0.96, top = 0.99, bottom = 0.15, hspace = 0.05, wspace = 0.05)


	for true_label, pred_label, true_axial, pred_axial, pred_prob, input_img in zip(y_true, y_pred, coord_trues, coord_preds, pred_probs, all_imgs):
		if true_label != pred_label and true_label == 0:
			cl_pt, cl_dir = axial_to_cl_point_ori(all_cls, all_cl_sums, pred_label, pred_axial * 120)
			up = arbitrary_perpendicular_vector(cl_dir)
			rgb, dep = get_depth_map(p, cl_pt, cl_dir, up, get_outputs = True)
			dep = transforms.GaussianBlur(21, 7)(torch.tensor(dep).unsqueeze(0)).squeeze().numpy()
			dep = (dep - dep.min()) / (dep.max() - dep.min())

			"""
			plt.subplot(1, 3, 1)
			plt.imshow(dep)
			plt.subplot(1, 3, 2)
			plt.imshow(rgb)
			plt.subplot(1, 3, 3)
			plt.imshow(input_img.reshape(input_img.shape[-2], input_img.shape[-1]))
			plt.suptitle("Predicted: {} Axial: {}, Orig axial: {}".format(pred_label, pred_axial, true_axial))
			plt.show()
			"""
			plt.imshow(input_img.squeeze(), vmin = 0, vmax = 1)
			plt.axis("off")
			plt.savefig("true-{}-{}.png".format(true_label, true_axial))
			plt.clf()
			plt.imshow(dep.squeeze(), vmin = 0, vmax = 1)
			plt.axis("off")
			plt.savefig("pred-{}-{}.png".format(pred_label, pred_axial))
			exit()



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

	"""
	plt.plot(y_pred, label = "Pred")
	plt.plot(y_true, label = "True")
	plt.title("Types")
	plt.show()
	
	plt.plot(coords_pred, label = "Pred")
	plt.plot(coords_true, label = "True")
	plt.title("Coords")
	plt.show()
	"""


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
