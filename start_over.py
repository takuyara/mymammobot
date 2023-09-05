import os
import torch
import argparse
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

"""
class ImageDataset(torch.utils.Dataset):
	def __init__(self, paths, transform_img):
		self.transform_img = transform_img
		for this_path in self.paths:
			for npy_path in os.listdir(this_path):
				if npy_path.endswith(".npy"):
					img = np.load(npy_path)
"""

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--base-path", type = str, default = "./")
	parser.add_argument("--train-split", type = str, nargs = "+", default = ["virtual_dataset/full/train"])
	parser.add_argument("--val-split", type = str, nargs = "+", default = ["real_dataset/confirmed/real-0", "real_dataset/confirmed/real-1", "real_dataset/confirmed/real-2"])
	parser.add_argument("-nw", "--num-workers", type = int, default = 0)
	parser.add_argument("--device", type = str, default = "cuda")
	parser.add_argument("--n-channels", type = int, default = 1)
	parser.add_argument("--num-classes", type = int, default = 3)
	parser.add_argument("--epochs", type = int, default = 30)
	parser.add_argument("--batch-size", type = int, default = 32)
	parser.add_argument("--lr", type = float, default = 1e-4)
	return parser.parse_args()

def get_transform(training, n_channels):
	def fun(img):
		img = torch.tensor(img).unsqueeze(0)
		if training:
			img = transforms.GaussianBlur(21, 7)(img)
		img = (img - img.min()) / (img.max() - img.min())
		return img.repeat(n_channels, 1, 1)
	return fun

def get_dataset(paths, transform_img):
	img_list, label_list = [], []
	for this_path in paths:
		for npy_path in tqdm(os.listdir(this_path)):
			if npy_path.endswith(".npy"):
				npy_path = os.path.join(this_path, npy_path)
				img = transform_img(np.load(npy_path))
				label = int(np.loadtxt(npy_path.replace(".npy", "_clbase.txt")).ravel()[0])
				img_list.append(img)
				label_list.append(torch.tensor(label))
	img_list, label_list = torch.stack(img_list, dim = 0), torch.stack(label_list, dim = 0)
	return TensorDataset(img_list, label_list)

def main():
	args = get_args()
	train_set = get_dataset([os.path.join(args.base_path, ts) for ts in args.train_split], get_transform(True, args.n_channels))
	print("Train load done.", flush = True)
	val_set = get_dataset([os.path.join(args.base_path, ts) for ts in args.val_split], get_transform(False, args.n_channels))
	print("Val load done.", flush = True)
	train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
	val_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
	model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
	if args.n_channels != 3:
		model.conv1 = nn.Conv2d(args.n_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
	model.fc = nn.Linear(model.fc.in_features, args.num_classes)
	model = model.to(args.device)
	optimiser = torch.optim.Adam(model.parameters(), lr = args.lr)
	for epoch in range(args.epochs):
		for phase, loader in [("train", train_loader), ("val", val_loader)]:
			if phase == "train":
				model.train()
			else:
				model.eval()
			y_true, y_pred = [], []
			sum_loss = num_loss = 0
			for imgs, labels in tqdm(loader):
				imgs, labels = imgs.to(args.device).float(), labels.to(args.device)
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
			print("Epoch {} phase {}: loss = {:.4f}, accuracy = {:.4f}, f1 = {:.4f}".format(epoch, phase, sum_loss / num_loss, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average = "macro")), flush = True)

if __name__ == '__main__':
	main()
