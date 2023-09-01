import argparse
hidden_arg_names = ["base_dir", "train_split", "val_split", "batch_size", "epochs", "output_dim", "img_size", "num_workers", "device", "save_path", "data_stats", "mesh_path"]

def get_base_parser(parser):
	parser.add_argument("--base-dir", type = str, default = "./depth-images", help = "The base dataset dir.")
	parser.add_argument("--train-split", type = str, default = "trains1.txt", help = "The path to train set split.")
	parser.add_argument("--val-split", type = str, default = "vals1.txt", help = "The path to val set split.")
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("-lr", "--learning-rate", type = float, default = 1e-3, help = "The learning rate.")
	parser.add_argument("-e", "--epochs", type = int, default = 20, help = "The number of epochs for training.")
	parser.add_argument("-b", "--batch-size", type = int, default = 32, help = "The batch size.")
	parser.add_argument("-od", "--output-dim", type = int, default = 6, help = "The output dimension to the fc layer.")
	parser.add_argument("-len", "--length", type = int, default = 3, help = "The length of history frames.")
	parser.add_argument("-spc", "--spacing", type = int, default = 5, help = "The spacing for history frames.")
	parser.add_argument("--img-size", type = int, default = 224, help = "The input image size.")
	parser.add_argument("-nw", "--num-workers", type = int, default = 4, help = "The number of workers.")
	parser.add_argument("-d", "--device", type = str, default = "cuda", help = "The GPU device name.")
	parser.add_argument("--save-path", type = str, default = "./checkpoints", help = "The base directory to save weights file.")
	parser.add_argument("--data-stats", type = str, default = "./data_stats.json", help = "The pose metadata file. Should contain mean and std for rotation and quaternion.")
	parser.add_argument("--loss-fun", type = str, default = "l2", choices = ["l1", "l2", "l2_trans", "balanced_l1"], help = "The loss function.")
	parser.add_argument("--hispose-noise", type = float, default = 0, help = "The noise added to previous poses.")
	parser.add_argument("--skip-prev-frame", action = "store_true", default = False, help = "Whether the history starts at i-1 or i-K when encoding i.")
	parser.add_argument("--model-sel-metric", type = str, choices = ["loss", "trans_err", "rot_err", "comb_err"], default = "comb_err", help = "The model selection metric.")
	parser.add_argument("--model-sel-rot-coef", type = float, default = 0.5, help = "The coefficient applied to rotation error for combined error metric.")
	parser.add_argument("--relative-coef", type = float, default = 1)
	parser.add_argument("--temporal-max", type = int, default = 10)
	parser.add_argument("--train-preprocess", type = str, default = "mesh")
	parser.add_argument("--val-preprocess", type = str, default = "sfs2mesh")
	parser.add_argument("--uses-tc", action = "store_true", default = False)
	parser.add_argument("--val-tc", action = "store_true", default = False)
	parser.add_argument("--train-gen", action = "store_true", default = False)
	parser.add_argument("--val-gen", action = "store_true", default = False)
	parser.add_argument("--n-channels", type = int, default = 1)
	return parser

def get_r3d_parser(parser):
	parser.add_argument("--pretrained-path", type = str, default = "./pretrained/r3d18_KM_200ep.pth", help = "The pretrained file path.")
	parser.add_argument("--model-depth", type = int, default = 18, help = "The depth of the resnet model.")
	parser.add_argument("--n-pretrained-classes", type = int, default = 1039, help = "The number of classes on the pretrained model.")
	return parser

def get_atloc_parser(parser):
	parser.add_argument("--atloc-base", type = str, default = "resnet34", choices = ["resnet18", "resnet34", "resnet50"])
	parser.add_argument("--dropout", type = float, default = 0.5, help = "The dropout rate for AtLoc.")
	parser.add_argument("--img-encode-dim", type = int, default = 2048, help = "The feature dim for AtLoc.")
	parser.add_argument("--uses-posenet", action = "store_true", default = False)
	return parser

def get_hisenc_parser(parser):
	parser.add_argument("--atloc-path", type = str, default = None, help = "The pretrained file path.")
	parser.add_argument("--hidden-size", type = int, default = 512, help = "The hidden size of LSTM.")
	parser.add_argument("--num-layers", type = int, default = 2, help = "The number of layers for LSTM.")
	parser.add_argument("--mlp-hisenc-out", type = int, nargs = "+", default = [512, 256], help = "The number of neurons for each MLP layer (excluding the last prediction one).")
	parser.add_argument("--bidirectional", action = "store_true", default = False, help = "Whether use bidirectional LSTM.")
	parser.add_argument("--img-his-only", action = "store_true", default = False, help = "Whether only encode images in history encoder.")
	return parser

def get_fusepred_parser(parser):
	parser.add_argument("--hisenc-path", type = str, default = None, help = "The path to the history encoder weights.")
	parser.add_argument("--mlp-branch-his", type = int, nargs = "+", default = [512, 256], help = "The MLP neurons for branch before fusion for history encoding.")
	parser.add_argument("--mlp-branch-cur", type = int, nargs = "+", default = [1024, 512, 256], help = "The MLP neurons for branch before fusion for image encoding.")
	parser.add_argument("--mlp-fusepred-out", type = int, nargs = "+", default = [512, 256, 128], help = "The MLP neurons for final prediction.")
	parser.add_argument("--fuse-mode", type = str, default = "cat", choices = ["cat", "plus"], help = "The fuse mode for 2 input features.")
	return parser

def get_finalsel_parser(parser):
	parser.add_argument("--fusepred-path", type = str, default = None, help = "The path to the trained fusepred path.")
	parser.add_argument("--mlp-weighting", type = int, nargs = "+", default = [512, 1024], help = "The weighting layer size.")
	parser.add_argument("--negative-weights", action = "store_true", default = False, help = "Whether allow negative weights (not using Softmax).")
	return parser

def get_test_parser(parser):
	parser.add_argument("--finalsel-path", type = str, default = None, help = "The path to the final selection path.")
	parser.add_argument("--test-split", type = str, default = "tests1.txt", help = "The path to test set (real images) split.")
	parser.add_argument("--test-modality", type = str, default = "SFS", choices = ["SFS", "mesh"], help = "The test modality.")
	parser.add_argument("--save-predictions", action = "store_true", default = False, help = "Whether save the predicted trajectory (as well as the ground truth) for plot.")
	parser.add_argument("--test-preprocess", type = str, default = "sfs2mesh", choices = ["sfs2mesh", "mesh2sfs", "sfs", "mesh", "quantile", "hist_simple", "hist_complex"])
	parser.add_argument("--test-gen", action = "store_true", default = False)
	parser.add_argument("--test-rotatable", action = "store_true", default = False)
	return parser

def get_args(*reqs):
	parser = argparse.ArgumentParser()
	parser = get_base_parser(parser)
	for req in reqs:
		if req == "r3d":
			parser = get_r3d_parser(parser)
		elif req == "atloc":
			parser = get_atloc_parser(parser)
		elif req == "hisenc":
			parser = get_hisenc_parser(parser)
		elif req == "fusepred":
			parser = get_fusepred_parser(parser)
		elif req == "finalsel":
			parser = get_finalsel_parser(parser)
		elif req == "test":
			parser = get_test_parser(parser)
	args = parser.parse_args()
	for arg_name, arg_value in vars(args).items():
		if arg_name not in hidden_arg_names:
			print(f"{arg_name}: {arg_value}")
	return args
