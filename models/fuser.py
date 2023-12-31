import torch
from torch import nn
from models.model_utils import get_mlp

class LSTMFuser(nn.Module):
	def __init__(self, length, f1_dim, f2_dim, hidden_size, num_layers, mlp_neurons, output_dim, dropout, bidirectional, f1_only):
		super(LSTMFuser, self).__init__()
		self.f1_only = f1_only
		self.lstm1 = nn.LSTM(f1_dim, hidden_size, num_layers, batch_first = True, dropout = dropout, bidirectional = bidirectional)
		if not self.f1_only:
			self.lstm2 = nn.LSTM(f2_dim, hidden_size, num_layers, batch_first = True, dropout = dropout, bidirectional = bidirectional)
			self.mlp, prev_features = get_mlp(2 * hidden_size * num_layers * (2 if bidirectional else 1), mlp_neurons, dropout)
		else:
			self.mlp, prev_features = get_mlp(hidden_size * num_layers * (2 if bidirectional else 1), mlp_neurons, dropout)
		self.fc = nn.Linear(prev_features, output_dim)

	def forward(self, x1, x2, get_encode = False, return_both = False):
		_, (x1, __) = self.lstm1(x1)
		x1 = x1.swapaxes(0, 1)
		x1 = x1.reshape(x1.size(0), -1)
		if not self.f1_only:
			_, (x2, __) = self.lstm2(x2)
			x2 = x2.swapaxes(0, 1)
			x2 = x2.reshape(x2.size(0), -1)
			xc = torch.cat((x1, x2), dim = -1)
		else:
			xc = x1
		xc = self.mlp(xc)
		out = self.fc(xc)
		if get_encode:
			if return_both:
				return xc, out
			else:
				return xc
		else:
			return out
