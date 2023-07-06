import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.arguments import get_args
from utils.file_utils import get_dir_list
from datasets.cl_dataset import CLDataset
from models.resnet import generate_model as generate_r3d

def train_val(model, dataloaders, optimiser, epochs, device):
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
				if b_id % 10 == 0:
					print(i, phase, b_id, sum_loss / num_loss)
			print("Epoch {} {} done. Avg loss = {.5f}".format(b_id, phase, sum_loss / num_loss))

def main():
	args = get_args("r3d")
	datasets = {phase : CLDataset(args.base_dir, get_dir_list(split_path), args.length, args.spacing,
		transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), None)
		for phase, split_path in [("train", args.train_split), ("val", args.val_split)]}
	dataloaders = {phase : DataLoader(ds, batch_size = args.batch_size,
		num_workers = args.num_workers, shuffle = phase == "train") for phase, ds in datasets.items()}
	device = torch.device(args.device)
	model = generate_r3d(args.model_depth, n_classes = args.n_pretrained_classes)
	model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])
	model.fc = nn.Linear(model.fc.in_features, args.output_dim)
	model = model.to(device)
	optimiser = optim.Adam(model.parameters(), lr = args.learning_rate)
	train_val(model, dataloaders, optimiser, args.epochs, device)

if __name__ == '__main__':
	main()
