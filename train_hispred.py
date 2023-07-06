import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
from copy import deepcopy

from utils.arguments import get_args
from utils.file_utils import get_dir_list
from utils.preprocess import get_img_transform
from datasets.cl_dataset import CLDataset
from models.atloc import AtLoc
from models.fuser import LSTMFuser
from models.fuse_predictor import MLPFusePredictor

def train_val(model_atloc, model_fuser, model_fuse_predictor, dataloaders, optimiser, epochs, device):
	model_atloc.eval()
	model_fuser.eval()
	min_loss = 1e100
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model_fuse_predictor.train()
			else:
				model_fuse_predictor.eval()
			sum_loss, num_loss = 0, 0
			for b_id, (imgs, poses) in enumerate(dataloaders[phase]):
				with torch.set_grad_enabled(phase == "train"):
					imgs, poses = imgs.to(device).float(), poses.to(device).float()
					#imgs_input, poses_input = imgs[ : , ]
					#b, s, c, w, h to b, c, s, w, h
					poses_history, poses_true = poses[ : , : -1, ...], poses[ : , -1, ...]
					bs, sqlen, C, W, H = tuple(imgs.shape)
					imgs = imgs.reshape(bs * sqlen, C, W, H)
					imgs_encode = model_atloc(imgs, get_encode = True)
					imgs_encode = imgs_encode.reshape(bs, sqlen, imgs_encode.size(1))
					history_encode, current_encode = imgs_encode[ : , : -1, : ], imgs_encode[ : , -1, : ]
					history_encode = model_fuser(history_encode, poses_history, get_encode = True)
					poses_pred = model_fuse_predictor(history_encode, current_encode)
					loss = nn.MSELoss()(poses_true, poses_pred)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				sum_loss += loss.item()
				num_loss += 1
			t_loss = sum_loss / num_loss
			print("Epoch {} {} done. Avg loss = {:.5f}".format(b_id, phase, t_loss), flush = True)
			if phase == "val" and t_loss < min_loss:
				min_loss, min_epoch, best_state_dict = t_loss, i, deepcopy(model_fuse_predictor.state_dict())
	print("Min loss {:.5f} occured at epoch {}".format(min_loss, min_epoch))
	return min_loss, min_epoch, best_state_dict

def main():
	args = get_args("hisenc", "hispred")
	datasets = {phase : CLDataset(args.base_dir, get_dir_list(split_path), args.length, args.spacing,
		get_img_transform(args), None) for phase, split_path in [("train", args.train_split), ("val", args.val_split)]}
	dataloaders = {phase : DataLoader(ds, batch_size = args.batch_size,
		num_workers = args.num_workers, shuffle = phase == "train") for phase, ds in datasets.items()}
	device = torch.device(args.device)
	model_atloc = AtLoc(models.resnet34(pretrained = False), droprate = args.dropout, feat_dim = args.feature_dim).to(device)
	model_atloc.load_state_dict(torch.load(args.pretrained_path)["model_state_dict"])
	model_fuser = LSTMFuser(args.length, args.feature_dim, args.output_dim, args.hidden_size, args.num_layers, args.mlp_neurons, args.output_dim, args.dropout).to(device)
	model_fuser.load_state_dict(torch.load(args.pretrained_hisenc_path))
	model_fuse_predictor = MLPFusePredictor(args.mlp_neurons[-1], args.feature_dim, args.mlp_branch_his, args.mlp_branch_cur, args.mlp_final, args.output_dim, args.dropout).to(device)
	optimiser = optim.Adam(model_fuse_predictor.parameters(), lr = args.learning_rate)
	min_loss, min_epoch, best_state_dict = train_val(model_atloc, model_fuser, model_fuse_predictor, dataloaders, optimiser, args.epochs, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"hisenc_fuse_predictor_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
