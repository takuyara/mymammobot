import torch
from torch import nn
import kornia.geometry as KG

lbd = 0.3

def corr_loss_img(inp, tgt, threshold = 0):
	inp, tgt = inp.flatten(), tgt.flatten()
	dt = torch.stack([inp[inp > threshold], tgt[inp > threshold]], axis = 0)
	return -torch.corrcoef(dt)[1][0]

def corr_loss(b_inp, b_tgt):
	losses = sum([corr_loss_img(inp, tgt) + lbd * torch.sum(inp == 0) for inp, tgt in zip(torch.unbind(b_inp), torch.unbind(b_tgt))])
	return losses

"""
def corr_loss_kornia(q, lbd):
	def loss_fun(inp, tgt, reduction = "none"):
		threshold = torch.quantile(inp, q).detach()
		loss_v = corr_loss_img(inp, tgt, threshold) + lbd * torch.sum(inp == 0)
		print(loss_v.item())
		return torch.ones_like(inp) * loss_v
	return loss_fun
"""

def corr_loss_kornia(inp, tgt, reduction = "none"):
	return torch.ones_like(inp) * corr_loss_img(inp, tgt)

class TransformRigidBody(nn.Module):
	def __init__(self, batch_size, height, width):
		super(TransformRigidBody, self).__init__()
		self.batch_size = batch_size
		self.translation = nn.Parameter(torch.zeros(batch_size, 2))
		self.rotation = nn.Parameter(torch.zeros(batch_size, 1))
	def get_translation_matrix(self):
		t = self.translation.reshape(-1, 2, 1)
		x1 = torch.eye(2).reshape(1, 2, 2).repeat(self.batch_size, 1, 1).to(t.device)
		x2 = torch.tensor([0, 0, 1]).reshape(1, 1, 3).repeat(self.batch_size, 1, 1).to(t.device)
		x1 = torch.cat([x1, t], dim = 2)
		x2 = torch.cat([x1, x2], dim = 1)
		#print("trans", x2, self.translation, x2.shape)
		return x2
	def array_2to3(self, t):
		x1 = torch.zeros(self.batch_size, 2, 1).to(t.device)
		x2 = torch.tensor([0, 0, 1]).reshape(1, 1, 3).repeat(self.batch_size, 1, 1).to(t.device)
		x1 = torch.cat([t, x1], dim = 2)
		x2 = torch.cat([x1, x2], dim = 1)
		return x2
	def get_rotation_shearing_matrix(self, t1):
		#print("rot/shear", t1, torch.cos(t1), torch.sin(t1))
		cost, sint = torch.cos(t1), torch.sin(t1)
		t = torch.stack([cost, sint, -sint, cost], dim = -1).reshape(-1, 2, 2)
		x2 = self.array_2to3(t)
		#print("rot/shear", x2, t1, x2.shape)
		return x2
	def get_6dof_matrix(self):
		T = self.get_translation_matrix()
		R = self.get_rotation_shearing_matrix(self.rotation)
		return R @ T
	def forward(self, x, inverse = False):
		trans_matrix = self.get_6dof_matrix()
		#print("6dof", trans_matrix)
		if inverse:
			trans_matrix = torch.inverse(trans_matrix)
		warper = KG.HomographyWarper(x.size(-2), x.size(-1)).to(x.device)
		return warper(x, trans_matrix)

class TransformAffine(nn.Module):
	def __init__(self, batch_size, height, width):
		super(TransformAffine, self).__init__()
		self.batch_size = batch_size
		self.translation = nn.Parameter(torch.zeros(batch_size, 2))
		self.rotation = nn.Parameter(torch.zeros(batch_size, 1))
		self.scaling = nn.Parameter(torch.ones(batch_size, 2))
		self.shearing = nn.Parameter(torch.zeros(batch_size, 1))
	def get_translation_matrix(self):
		t = self.translation.reshape(-1, 2, 1)
		x1 = torch.eye(2).reshape(1, 2, 2).repeat(self.batch_size, 1, 1).to(t.device)
		x2 = torch.tensor([0, 0, 1]).reshape(1, 1, 3).repeat(self.batch_size, 1, 1).to(t.device)
		x1 = torch.cat([x1, t], dim = 2)
		x2 = torch.cat([x1, x2], dim = 1)
		#print("trans", x2, self.translation, x2.shape)
		return x2
	def array_2to3(self, t):
		x1 = torch.zeros(self.batch_size, 2, 1).to(t.device)
		x2 = torch.tensor([0, 0, 1]).reshape(1, 1, 3).repeat(self.batch_size, 1, 1).to(t.device)
		x1 = torch.cat([t, x1], dim = 2)
		x2 = torch.cat([x1, x2], dim = 1)
		return x2
	def get_rotation_shearing_matrix(self, t1):
		#print("rot/shear", t1, torch.cos(t1), torch.sin(t1))
		cost, sint = torch.cos(t1), torch.sin(t1)
		t = torch.stack([cost, sint, -sint, cost], dim = -1).reshape(-1, 2, 2)
		x2 = self.array_2to3(t)
		#print("rot/shear", x2, t1, x2.shape)
		return x2
	def get_scaling_matrix(self):
		t = torch.diag_embed(self.scaling)
		#print(t.shape)
		x2 = self.array_2to3(t)
		#print("scal", x2, self.scaling, x2.shape)
		return x2
	def get_6dof_matrix(self):
		T = self.get_translation_matrix()
		S = self.get_scaling_matrix()
		R = self.get_rotation_shearing_matrix(self.rotation)
		H = self.get_rotation_shearing_matrix(self.shearing)
		return H @ S @ torch.inverse(H) @ R @ T
	def forward(self, x, inverse = False):
		trans_matrix = self.get_6dof_matrix()
		#print("6dof", trans_matrix)
		if inverse:
			trans_matrix = torch.inverse(trans_matrix)
		warper = KG.HomographyWarper(x.size(-2), x.size(-1)).to(x.device)
		return warper(x, trans_matrix)

class ImageRegistrator(nn.Module):
	def __init__(self, batch_size, height, width, pyramid_levels = 5, lr = 1e-3, num_iters = 500, tolerance = 1e-4, loss_fn = corr_loss, transform_type = "rigid"):
		super(ImageRegistrator, self).__init__()
		if transform_type == "rigid":
			self.transform = TransformRigidBody(batch_size, height, width)
		elif transform_type == "affine":
			self.transform = TransformAffine(batch_size, height, width)
		else:
			raise NotImplementedError
		self.pyramid_levels = pyramid_levels
		self.num_iters = num_iters
		self.loss_fn = loss_fn
		self.tolerance = tolerance
		self.optimiser = torch.optim.Adam(self.transform.parameters(), lr = lr)
	def forward(self, src, tgt, return_loss = False):
		img_src_pyr = KG.build_pyramid(src, self.pyramid_levels)[::-1]
		img_tgt_pyr = KG.build_pyramid(tgt, self.pyramid_levels)[::-1]
		all_losses = []
		for img_src, img_tgt in zip(img_src_pyr, img_tgt_pyr):
			prev_loss = 1e10
			for i in range(self.num_iters):
				self.optimiser.zero_grad()
				with torch.enable_grad():
					loss = self.loss_fn(self.transform(img_src), img_tgt)
					loss += self.loss_fn(self.transform(img_tgt, inverse = True), img_src)
				cur_loss = loss.mean().item()
				if abs(cur_loss - prev_loss) < self.tolerance:
					break
				prev_loss = cur_loss
				loss.backward()
				self.optimiser.step()
				all_losses.append(cur_loss / 2 / img_src.size(0))
		if not return_loss:
			return self.transform(src)
		else:
			return self.transform(src), all_losses
