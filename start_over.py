import os
import torch
import argparse
from torch import nn
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

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
	parser.add_argument("--model-type", type = str, default = "resnet")
	parser.add_argument("--dropout", type = float, default = 0.4)
	parser.add_argument("--binary", action = "store_true", default = False)
	parser.add_argument("--cap", type = float, default = 100)
	return parser.parse_args()


max_hists = [[], [], []]

def get_transform(training, n_channels, cap):
	def fun(img):
		img = torch.tensor(img).unsqueeze(0)
		if training:
			img = transforms.GaussianBlur(21, 7)(img)
			img = torch.minimum(img, torch.tensor(cap))
		img = transforms.Resize(100)(img)
		img = (img - img.min()) / (img.max() - img.min())
		return img.repeat(n_channels, 1, 1)
	return fun

class PreloadDataset(Dataset):
	def __init__(self, img_path, label_path, transform, binary):
		self.img_data = np.load(img_path)
		self.label_data = np.load(label_path)
		self.transform = transform
		self.binary = binary
		"""
		for i in range(len(self.img_data)):
			max_hists[self.label_data[i, ...]].append(self.img_data[i, ...].max())
		"""

	def __getitem__(self, idx):
		lb = self.label_data[idx, ...]
		img = self.transform(self.img_data[idx, ...])
		if self.binary:
			lb = 0 if lb == 0 else 1
			if lb == 0:
				img = transforms.functional.adjust_gamma(img, 0.6)
		return img, lb
	def __len__(self):
		return len(self.img_data)

def main():
	args = get_args()
	print(args)
	if args.binary:
		args.num_classes = 2
	train_set = PreloadDataset(os.path.join(args.base_path, f"{args.train_path}_img.npy"), os.path.join(args.base_path, f"{args.train_path}_label.npy"), get_transform(True, args.n_channels, args.cap), args.binary)
	"""
	plt.subplot(1, 3, 1)
	plt.hist(max_hists[0], bins = 20)
	plt.subplot(1, 3, 2)
	plt.hist(max_hists[1], bins = 20)
	plt.subplot(1, 3, 3)
	plt.hist(max_hists[2], bins = 20)
	plt.show()
	exit()
	"""
	print("Train load done.", flush = True)
	val_set = PreloadDataset(os.path.join(args.base_path, f"{args.val_path}_img.npy"), os.path.join(args.base_path, f"{args.val_path}_label.npy"), get_transform(False, args.n_channels, args.cap), args.binary)
	print("Val load done.", flush = True)
	train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
	val_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
	if args.model_type == "resnet":
		model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
		if args.n_channels != 3:
			model.conv1 = nn.Conv2d(args.n_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
		model.fc = nn.Linear(model.fc.in_features, args.num_classes)
	else:
		model = models.swin_t(weights = models.Swin_T_Weights.DEFAULT, dropout = args.dropout)
		if args.n_channels != 3:
			model.features[0][0] = nn.Conv2d(args.n_channels, 96, kernel_size = 4, stride = 4)
		model.head = nn.Linear(model.head.in_features, args.num_classes)
	model = model.to(args.device)
	optimiser = torch.optim.Adam(model.parameters(), lr = args.lr)
	max_acc = 0
	for epoch in range(args.epochs):
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
				max_acc, best_weights = acc, deepcopy(model.state_dict())
	torch.save(best_weights, os.path.join(args.save_path, f"ckpt-{max_acc:.4f}.pt"))

if __name__ == '__main__':
	main()
