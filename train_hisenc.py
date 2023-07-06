import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from utils.arguments import get_args
from utils.file_utils import get_dir_list
from datasets.cl_dataset import CLDataset
from models.atloc import AtLoc
from models.fuser import LSTMFuser

def train_val(model_atloc, model_fuser, dataloaders, optimiser, epochs, device):
	model_atloc.eval()
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model_fuser.train()
			else:
				model_fuser.eval()
			sum_loss, num_loss = 0, 0
			for b_id, (imgs, poses) in enumerate(dataloaders[phase]):
				with torch.set_grad_enabled(phase == "train"):
					imgs, poses = imgs.to(device).float(), poses.to(device).float()
					#imgs_input, poses_input = imgs[ : , ]
					#b, s, c, w, h to b, c, s, w, h
					imgs_history, poses_history = imgs[ : , : -1, ...], poses[ : , : -1, ...]
					poses_true = poses[ : , -1, ...]
					bs, sqlen, C, W, H = tuple(imgs_history.shape)
					imgs_history = imgs_history.reshape(bs * sqlen, C, W, H)
					imgs_history = model_atloc(imgs_history, get_encode = True)
					imgs_history = imgs_history.reshape(bs, sqlen, imgs_history.size(1))
					poses_pred = model_fuser(imgs_history, poses_history)
					loss = nn.MSELoss()(poses_true, poses_pred)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				sum_loss += loss.item()
				num_loss += 1
				if b_id % 50 == 0:
					print(i, phase, b_id, sum_loss / num_loss)
			print("Epoch {} {} done. Avg loss = {.5f}".format(b_id, phase, sum_loss / num_loss))

def main():
	args = get_args("hisenc")
	datasets = {phase : CLDataset(args.base_dir, get_dir_list(split_path), args.length, args.spacing,
		transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), None)
		for phase, split_path in [("train", args.train_split), ("val", args.val_split)]}
	dataloaders = {phase : DataLoader(ds, batch_size = args.batch_size,
		num_workers = args.num_workers, shuffle = phase == "train") for phase, ds in datasets.items()}
	device = torch.device(args.device)
	model_atloc = AtLoc(models.resnet34(pretrained = False), droprate = args.dropout, feat_dim = args.feature_dim).to(device)
	model_atloc.load_state_dict(torch.load(args.pretrained_path)["model_state_dict"])
	model_fuser = LSTMFuser(args.length, args.feature_dim, args.output_dim, args.hidden_size, args.num_layers, args.mlp_neurons, args.output_dim, args.dropout).to(device)
	optimiser = optim.Adam(model_fuser.parameters(), lr = args.learning_rate)
	train_val(model_atloc, model_fuser, dataloaders, optimiser, args.epochs, device)

if __name__ == '__main__':
	main()
