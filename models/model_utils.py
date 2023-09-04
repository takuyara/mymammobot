from torch import nn

def get_mlp(in_dim, num_neurons, dropout):
	module_list = []
	for this_neurons in num_neurons:
		this_layer = nn.Sequential(nn.Linear(in_dim, this_neurons), nn.LeakyReLU(0.2), nn.Dropout(dropout))
		module_list.append(this_layer)
		in_dim = this_neurons
	return nn.Sequential(*module_list), in_dim
