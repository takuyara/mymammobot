import torch
from torch.utils.data import TensorDataset, DataLoader

def get_processed_dataloaders(dataloaders, device, *models):
	def get_a_loader(this_loader, shuffle):
		if len(models) == 0:
			return this_loader
		n_model_to_n_res = {1 : 3, 2 : 3, 3 : 5}
		all_res = [[] for __ in range(n_model_to_n_res[len(models)])]
		for b_id, (imgs, poses) in enumerate(this_loader):
			with torch.no_grad():
				imgs, poses = imgs.to(device).float(), poses.to(device).float()
				#b, s, c, w, h 
				poses_history, poses_true = poses[ : , : -1, ...], poses[ : , -1, ...]
				bs, sqlen, C, W, H = tuple(imgs.shape)
				imgs = imgs.reshape(bs * sqlen, C, W, H)
				model_atloc = models[0]
				imgs_encode, atloc_pred = model_atloc(imgs, get_encode = True, return_both = True)
				imgs_encode = imgs_encode.reshape(bs, sqlen, imgs_encode.size(1))
				atloc_pred = atloc_pred.reshape(bs, sqlen, atloc_pred.size(1))[ : , -1, : ]
				history_encode, current_encode = imgs_encode[ : , : -1, : ], imgs_encode[ : , -1, : ]
				if len(models) == 1:
					for i, this_res in enumerate([history_encode, poses_history, poses_true]):
						all_res[i].append(this_res.cpu())
					continue
				model_fuser = models[1]
				history_encode, fuser_pred = model_fuser(history_encode, poses_history, get_encode = True, return_both = True)
				if len(models) == 2:
					for i, this_res in enumerate([history_encode, current_encode, poses_true]):
						all_res[i].append(this_res.cpu())
					continue
				model_fuse_predictor = models[2]
				fusepred_pred = model_fuse_predictor(history_encode, current_encode)
				if len(models) == 3:
					for i, this_res in enumerate([current_encode, atloc_pred, fuser_pred, fusepred_pred, poses_true]):
						all_res[i].append(this_res.cpu())
					continue
				raise NotImplementedError
		all_res = [torch.cat(this_res, dim = 0) for this_res in all_res]
		dataset = TensorDataset(*all_res)
		dataloader = DataLoader(dataset, num_workers = this_loader.num_workers, shuffle = shuffle, batch_size = this_loader.batch_size)

	for t_model in models:
		t_model.eval()
	if type(dataloaders) == dict:
		return {phase : get_a_loader(this_loader, phase == "train") for phase, this_loader in dataloaders.items()}
	else:
		return get_a_loader(dataloader, False)
