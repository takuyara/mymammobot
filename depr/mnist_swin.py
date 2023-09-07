import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision import models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

loaders = {}
for phase in ["train", "val"]:
	dset = MNIST("./tmp", train = phase == "train", download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))]))
	loaders[phase] = DataLoader(dset, batch_size = 32, shuffle = phase == "train")


device = "cuda"
model = models.swin_v2_t(weights = models.Swin_V2_T_Weights.DEFAULT, dropout = 0.4)
model.features[0][0] = nn.Conv2d(1, 96, kernel_size = 4, stride = 4)
model.head = nn.Linear(model.head.in_features, 10)
model = model.to(device)
optimiser = torch.optim.Adam(model.parameters(), lr = 1e-4)

for epoch in range(30):
	for phase in ["train", "val"]:
		if phase == "train":
			model.train()
		else:
			model.eval()
		y_true, y_pred = [], []
		sum_loss = num_loss = 0
		for b_id, (imgs, labels) in enumerate(tqdm(loaders[phase])):
			if b_id > 500:
				continue
			imgs, labels = imgs.to(device), labels.to(device)
			with torch.set_grad_enabled(phase == "train"):
				logits = model(imgs)
				loss = nn.CrossEntropyLoss()(logits, labels)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
			y_true.append(labels.cpu().numpy())
			y_pred.append(torch.argmax(logits, dim = -1).cpu().numpy())
			sum_loss += loss.item() * imgs.size(0)
			num_loss += imgs.size(0)
		y_true, y_pred = np.concatenate(y_true, axis = 0), np.concatenate(y_pred, axis = 0)
		print(f"Epoch {epoch} {phase} loss = {sum_loss / num_loss :.4f}, acc = {accuracy_score(y_true, y_pred)}.")

