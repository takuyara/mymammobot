import torch
from torch import optim
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.nn_utils import get_loaders_loss_metrics, get_models
from utils.forward_passes import get_processed_dataloaders

def train_val(model_sel, dataloaders, optimiser, epochs, loss_fun, metric_template, device):
	for i in range(epochs):
		for phase in ["train", "val"]:
			if phase == "train":
				model_sel.train()
			else:
				model_sel.eval()
			this_metric = metric_template.new_copy()
			for b_id, (current_encode, atloc_pred, fuser_pred, fusepred_pred, poses_true) in enumerate(dataloaders[phase]):
				current_encode, atloc_pred, fuser_pred = current_encode.to(device), atloc_pred.to(device), fuser_pred.to(device)
				fusepred_pred, poses_true = fusepred_pred.to(device), poses_true.to(device)
				with torch.set_grad_enabled(phase == "train"):	
					sel_pred = model_sel(current_encode, atloc_pred, fuser_pred, fusepred_pred)
					loss = loss_fun(poses_true, sel_pred)
				if phase == "train":
					optimiser.zero_grad()
					loss.backward()
					optimiser.step()
				this_metric.add_batch(poses_true, sel_pred)
			print("Epoch {} {} done. Metrics: {}".format(i, phase, this_metric), flush = True)
			if phase == "val" and (i == 0 or this_metric.main_metric() < best_metric.main_metric()):
				best_metric, min_epoch, best_state_dict = this_metric, i, deepcopy(model_sel.state_dict())
	print("Best metric {} occured at epoch {}".format(best_metric, min_epoch))
	return best_metric.loss(), min_epoch, best_state_dict

def main():
	args = get_args("atloc", "hisenc", "fusepred", "finalsel")
	dataloaders, loss_fun, metric_template = get_loaders_loss_metrics(args)
	model_atloc, model_fuser, model_fuse_predictor, model_sel, device = get_models(args, "atloc+", "hisenc+", "fusepred+", "finalsel")
	optimiser = optim.Adam(model_sel.parameters(), lr = args.learning_rate)
	dataloaders = get_processed_dataloaders(dataloaders, device, model_atloc, model_fuser, model_fuse_predictor)
	min_loss, min_epoch, best_state_dict = train_val(model_sel, dataloaders, optimiser, args.epochs, loss_fun, metric_template, device)
	torch.save(best_state_dict, os.path.join(args.save_path, f"selector_{min_loss:.5f}.pth"))

if __name__ == '__main__':
	main()
