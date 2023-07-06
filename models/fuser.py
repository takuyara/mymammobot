import torch
from torch import nn

class LSTMFuser(nn.Module):
	def __init__(self, length, f1_dim, f2_dim, hidden_size, num_layers, mlp_neurons, output_dim, dropout):
		super(LSTMFuser, self).__init__()
		self.lstm1 = nn.LSTM(f1_dim, hidden_size, num_layers, batch_first = True, dropout = dropout)
		self.lstm2 = nn.LSTM(f2_dim, hidden_size, num_layers, batch_first = True, dropout = dropout)
		prev_features = 2 * hidden_size * num_layers
		module_list = []
		for this_features in mlp_neurons:
			this_layer = nn.Sequential(nn.Linear(prev_features, this_features), nn.ReLU(), nn.Dropout(dropout))
			module_list.append(this_layer)
			prev_features = this_features
		self.mlp = nn.Sequential(*module_list)
		self.fc = nn.Linear(prev_features, output_dim)

	def forward(self, x1, x2, get_encode = False):
		_, (x1, __) = self.lstm1(x1)
		_, (x2, __) = self.lstm2(x2)
		x1, x2 = x1.swapaxes(0, 1), x2.swapaxes(0, 1)
		x1, x2 = x1.reshape(x1.size(0), -1), x2.reshape(x2.size(0), -1)
		xc = torch.cat((x1, x2), dim = -1)
		xc = self.mlp(xc)
		out = self.fc(xc)
		if get_encode:
			return xc
		else:
			return out
