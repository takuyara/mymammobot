import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.file_utils import get_dir_list
from utils.preprocess import get_img_transform
from datasets.cl_dataset import CLDataset
from models.resnet import generate_model as generate_r3d

def train_val(model, dataloaders, optimiser, epochs, device):
	min_loss = 1e100
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model.train()
			else:
				model.eval()
			sum_loss, num_loss = 0, 0
			for b_id, (imgs, poses) in enumerate(dataloaders[phase]):
				with torch.set_grad_enabled(phase == "train"):
					imgs, poses = imgs.to(device).float(), poses.to(device).float()
					#b, s, c, w, h to b, c, s, w, h
					imgs = imgs.swapaxes(1, 2)
					poses = poses[ : , -1, ...]
					poses_ = model(imgs)
					loss = nn.MSELoss()(poses, poses_)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				sum_loss += loss.item()
				num_loss += 1
			t_loss = sum_loss / num_loss
			print("Epoch {} {} done. Avg loss = {:.5f}".format(b_id, phase, t_loss). flush = True)
			if phase == "val" and t_loss < min_loss:
				min_loss, min_epoch, best_state_dict = t_loss, i, deepcopy(model.state_dict())
	print("Min loss {:.5f} occured at epoch {}".format(min_loss, min_epoch))
	return min_loss, min_epoch, best_state_dict

def main():
	args = get_args("r3d")
	datasets = {phase : CLDataset(args.base_dir, get_dir_list(split_path), args.length, args.spacing,
		get_img_transform(args), None) for phase, split_path in [("train", args.train_split), ("val", args.val_split)]}
	dataloaders = {phase : DataLoader(ds, batch_size = args.batch_size,
		num_workers = args.num_workers, shuffle = phase == "train") for phase, ds in datasets.items()}
	device = torch.device(args.device)
	model = generate_r3d(args.model_depth, n_classes = args.n_pretrained_classes)
	model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])
	model.fc = nn.Linear(model.fc.in_features, args.output_dim)
	model = model.to(device)
	optimiser = optim.Adam(model.parameters(), lr = args.learning_rate)
	min_loss, min_epoch, best_state_dict = train_val(model, dataloaders, optimiser, args.epochs, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"r3d_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
