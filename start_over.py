import os
import time
import torch
import argparse
from torch import nn
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from utils.misc import randu_gen

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
		return img.repeat(args.n_channels, 1, 1)
	return fun

class PreloadDataset(Dataset):
	def __init__(self, img_path, label_path, transform, args):
		self.img_data = np.load(img_path)
		self.label_data = np.load(label_path)
		self.transform = transform
		self.four_fold = args.four_fold
		self.four_thres = args.four_thres
		"""
		for i in range(len(self.img_data)):
			max_hists[self.label_data[i, ...]].append(self.img_data[i, ...].max())
		"""

	def __getitem__(self, idx):
		lb = int(self.label_data[idx, 0])
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

		return img, lb
	def __len__(self):
		return len(self.img_data)

def get_model(args):
	if args.model_type.find("resnet") != -1:
		if args.model_type == "resnet34":
			model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
		elif args.model_type == "resnet18":
			model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
		else:
			model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
		if args.n_channels != 3:
			model.conv1 = nn.Conv2d(args.n_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
		model.fc = nn.Linear(model.fc.in_features, args.num_classes)
	else:
		model = models.swin_v2_t(weights = models.Swin_V2_T_Weights.DEFAULT, dropout = args.dropout)
		if args.n_channels != 3:
			model.features[0][0] = nn.Conv2d(args.n_channels, 96, kernel_size = 4, stride = 4)
		model.head = nn.Linear(model.head.in_features, args.num_classes)
	return model

def main():
	args = get_args()
	print(args)
	if args.binary:
		args.num_classes = 2
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
	optimiser = torch.optim.Adam(model.parameters(), lr = args.lr)
	max_acc = 0
	for epoch in range(args.epochs):
		st_time = time.time()
		for phase, loader in [("train", train_loader), ("val", val_loader)]:
			if phase == "train":
				model.train()
			else:
				model.eval()
			y_true, y_pred = [], []
			sum_loss = num_loss = 0
			for imgs, labels in loader:
				imgs, labels = imgs.to(args.device).float(), labels.to(args.device).long()
				with torch.set_grad_enabled(phase == "train"):
					logits = model(imgs)
					loss = nn.CrossEntropyLoss()(logits, labels)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				sum_loss += loss.item() * imgs.size(0)
				num_loss += imgs.size(0)
				preds = torch.argmax(logits, dim = -1)
				y_true.append(labels.cpu().numpy())
				y_pred.append(preds.cpu().numpy())
			y_true, y_pred = np.concatenate(y_true, axis = 0), np.concatenate(y_pred, axis = 0)
			loss, acc, f1 = sum_loss / num_loss, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average = "macro")
			print("Epoch {} phase {}: loss = {:.4f}, accuracy = {:.4f}, f1 = {:.4f}".format(epoch, phase, loss, acc, f1), flush = True)
			if phase == "val" and acc > max_acc:
				max_acc, max_f1, best_weights = acc, f1, deepcopy(model.state_dict())
				if max_acc > 0.6:
					torch.save(best_weights, os.path.join(args.save_path, f"ckpt-{max_acc:.4f}.pt"))
		print("Epoch time: {:.2f} mins".format((time.time() - st_time) / 60))
	print("Max: ", max_acc, max_f1)

if __name__ == '__main__':
	main()
