import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from utils.arguments import get_args
from utils.file_utils import get_dir_list
from datasets.cl_dataset import CLDataset
from models.atloc import AtLoc

def train_val(model, dataloaders, optimiser, epochs, device):
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model.train()
			else:
				model.eval()
			for imgs, poses in dataloaders[phase]:
				with torch.set_grad_enabled(phase == "train"):
					print(imgs.shape, poses.shape)
					imgs, poses = imgs.to(device), poses.to(device)
					#imgs_input, poses_input = imgs[ : , ]
					#b, c, s, w, h to b, c, s, w, h

def main():
	args = get_args("hisenc")
	datasets = {phase : CLDataset(args.base_dir, get_dir_list(split_path), args.length, args.spacing,
		transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), None)
		for phase, split_path in [("train", args.train_split), ("val", args.val_split)]}
	dataloaders = {phase : DataLoader(ds, batch_size = args.batch_size,
		num_workers = args.num_workers, shuffle = phase == "train") for phase, ds in datasets.items()}
	device = torch.device(args.device)
	model = AtLoc(models.resnet34(pretrained = False), droprate = args.dropout, feat_dim = args.feature_dim)
	model.load_state_dict(torch.load(args.pretrained_path))
	optimiser = optim.Adam(model.parameters(), lr = args.learning_rate)
	train_val(model, dataloaders, optimiser, args.epochs, device)

if __name__ == '__main__':
	main()
