import torch
import kornia.geometry as KG
from domain_transfer.registration import ImageRegistrator, corr_loss_kornia

"""
def reg_depth_maps(src, tgt, device = "cuda:0", **kwargs):
	original_shape = src.shape
	src, tgt = torch.tensor(src).unsqueeze(-3), torch.tensor(tgt).unsqueeze(-3)
	if len(src.shape) == 3:
		src, tgt = src.unsqueeze(0), tgt.unsqueeze(0)
		batched = False
	else:
		batched = True
	print(src.shape, tgt.shape)
	src, tgt = src.to(device).float(), tgt.to(device).float()
	ir = ImageRegistrator(src.shape[0], src.shape[1], src.shape[2], **kwargs).to(device)
	src_ = ir(src, tgt).detach().cpu().numpy().reshape(original_shape)
	return src_

"""

ir_params = {"num_iterations": 1000, "lr": 1e-3, "tolerance": 1e-5}

def reg_depth_maps(src, tgt, device = "cuda:0"):
	original_shape = src.shape
	src, tgt = torch.tensor(src).unsqueeze(-3), torch.tensor(tgt).unsqueeze(-3)
	if len(src.shape) == 3:
		src, tgt = src.unsqueeze(0), tgt.unsqueeze(0)
	src, tgt = src.to(device).float(), tgt.to(device).float()
	registrator = KG.ImageRegistrator(KG.Similarity(True, False, True), warper = KG.HomographyWarper, loss_fn = corr_loss_kornia(0.7, 0), **ir_params).to(device)
	homo_transform = registrator.register(src, tgt)
	src_ = registrator.warp_src_into_dst(src).detach().cpu().numpy().reshape(original_shape)
	return src_
