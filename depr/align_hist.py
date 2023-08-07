import torch
from torch import nn
import numpy as np

lr = 1e-2
tol = 1e-5

class IntensityTransformer(nn.Module):
	def __init__(self, bins):
		super(IntensityTransformer, self).__init__()
		self.w = nn.Parameter(torch.ones(1))
		self.b = nn.Parameter(torch.zeros(1))
		self.bins = bins
	def forward(self, x, hy):
		x = x * self.w + self.b
		#cdf = torch.cat([torch.sum(x < i).reshape(-1) for i in self.bins])
		#cdf = torch.cat([torch.sum(x < i).reshape(-1) for i in self.bins])
		#pdf = cdf - torch.cat([torch.tensor(0., requires_grad = True).reshape(-1), cdf[ : -1]])
		#return nn.KLDivLoss(reduction = "batchmean")(cdf, hy)
		return nn.MSELoss()(torch.sum(torch.le(x, 140)), torch.sum(hy))

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

def main():
	em_values, ct_hist = np.load("em_values.npy"), np.load("ct_hist.npy")
	em_values, ct_hist = torch.tensor(em_values).float(), torch.tensor(ct_hist).float()
	trans = IntensityTransformer(torch.arange(len(ct_hist)) + 1)
	optimiser = torch.optim.Adam(trans.parameters(), lr = lr)
	hist_x = torch.arange(len(ct_hist)).float()
	prev_loss = 1e10
	while True:
		optimiser.zero_grad()
		loss = trans(em_values, ct_hist)
		loss.backward()
		print(trans.w.grad, trans.b.grad)
		optimiser.step()
		getBack(loss.grad_fn)
		loss = loss.item()
		if abs(loss - prev_loss) < tol:
			print(loss, prev_loss, abs(loss - prev_loss))
			break
		prev_loss = loss
		print(loss)
	print(trans.w, trans.b)
	"""
	plt.subplot(1, 2, 1)
	plt.bar((hist_x * trans.w + trans.b).numpy(), em_hist)
	plt.xlim(0, 256)
	plt.ylim(0, 0.04)
	plt.title("Adjusted EM plot")
	plt.subplot(1, 2, 2)
	plt.bar(hist_x.numpy(), ct_hist)
	plt.xlim(0, 256)
	plt.ylim(0, 0.04)
	plt.title("Adjusted CT plot")
	plt.show()
	"""

if __name__ == '__main__':
	main()
