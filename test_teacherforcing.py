import torch
from torch import optim
import os
from copy import deepcopy

from utils.arguments import get_args
from utils.nn_utils import get_loaders_loss_metrics, get_models

def train_val(model_atloc, model_fuser, model_fuse_predictor, model_sel, dataloader, metric_template, val_models, device):
	model_atloc.eval()
	model_fuser.eval()
	model_fuse_predictor.eval()
	model_sel.eval()
	atloc_metric = metric_template.new_copy()
	hisenc_metric = metric_template.new_copy()
	fusepred_metric = metric_template.new_copy()
	finalsel_metric = metric_template.new_copy()
	for b_id, (imgs, poses) in enumerate(dataloader):
		with torch.no_grad():
			if val_models < 1:
				continue
			imgs, poses = imgs.to(device).float(), poses.to(device).float()
			#imgs_input, poses_input = imgs[ : , ]
			#b, s, c, w, h to b, c, s, w, h
			poses_history, poses_true = poses[ : , : -1, ...], poses[ : , -1, ...]
			bs, sqlen, C, W, H = tuple(imgs.shape)
			imgs = imgs.reshape(bs * sqlen, C, W, H)
			imgs_encode, atloc_pred = model_atloc(imgs, get_encode = True, return_both = True)
			imgs_encode = imgs_encode.reshape(bs, sqlen, imgs_encode.size(1))
			atloc_pred = atloc_pred.reshape(bs, sqlen, atloc_pred.size(1))[ : , -1, : ]
			atloc_metric.add_batch(atloc_pred, poses_true)
			if val_models < 2:
				continue
			history_encode, current_encode = imgs_encode[ : , : -1, : ], imgs_encode[ : , -1, : ]
			history_encode, fuser_pred = model_fuser(history_encode, poses_history, get_encode = True, return_both = True)
			hisenc_metric.add_batch(fuser_pred, poses_true)
			if val_models < 3:
				continue
			fusepred_pred = model_fuse_predictor(history_encode, current_encode)
			fusepred_metric.add_batch(fusepred_pred, poses_true)
			if val_models < 4:
				continue
			current_encode, atloc_pred, fuser_pred, fusepred_pred = current_encode.detach(), atloc_pred.detach(), fuser_pred.detach(), fusepred_pred.detach()
			sel_pred = model_sel(current_encode, atloc_pred, fuser_pred, fusepred_pred)
			finalsel_metric.add_batch(sel_pred, poses_true)
	all_val = [("AtLoc", atloc_metric),("HisEnc", hisenc_metric), ("FusePred", fusepred_metric), ("FinalSel", finalsel_metric)]
	return all_val[ : val_models]

def main():
	args = get_args("atloc", "hisenc", "fusepred", "finalsel", "test")
	dataloader, loss_fun, metric_template = get_loaders_loss_metrics(args, test = True, test_set = False)
	model_atloc, model_fuser, model_fuse_predictor, model_sel, device = get_models(args, "atloc", "hisenc", "fusepred", "finalsel")
	val_models = 0
	if args.atloc_path is not None:
		model_atloc.load_state_dict(torch.load(args.atloc_path))
		val_models = 1
	if args.hisenc_path is not None and val_models == 1:
		model_fuser.load_state_dict(torch.load(args.hisenc_path))
		val_models = 2
	if args.fusepred_path is not None and val_models == 2:
		model_fuse_predictor.load_state_dict(torch.load(args.fusepred_path))
		val_models = 3
	if args.finalsel_path is not None and val_models == 3:
		model_sel.load_state_dict(torch.load(args.finalsel_path))
		val_models = 4
	val_results = train_val(model_atloc, model_fuser, model_fuse_predictor, model_sel, dataloader, metric_template, val_models, device)
	for model_name, results in val_results:
		print(model_name, results)

if __name__ == '__main__':
	main()

