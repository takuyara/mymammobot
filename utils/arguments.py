import argparse
def get_base_parser(parser):
	parser.add_argument("--base-dir", type = str, default = "E:\\nn-data\\MAMMOBOT\\data\\7Scenes\\CTCLDepthfiles", help = "The base dataset dir.")
	parser.add_argument("--train-split", type = str, default = "E:\\nn-data\\MAMMOBOT\\data\\7Scenes\\CTCLDepthfiles\\trains1.txt", help = "The path to train set split.")
	parser.add_argument("--val-split", type = str, default = "E:\\nn-data\\MAMMOBOT\\data\\7Scenes\\CTCLDepthfiles\\tests1.txt", help = "The path to val set split.")
	parser.add_argument("-lr", "--learning-rate", type = float, default = 1e-4, help = "The learning rate.")
	parser.add_argument("-e", "--epochs", type = int, default = 20, help = "The number of epochs for training.")
	parser.add_argument("-b", "--batch-size", type = int, default = 32, help = "The batch size.")
	parser.add_argument("-od", "--output-dim", type = int, default = 6, help = "The output dimension to the fc layer.")
	parser.add_argument("-len", "--length", type = int, default = 3, help = "The length of history frames.")
	parser.add_argument("-spc", "--spacing", type = int, default = 5, help = "The spacing for history frames.")
	parser.add_argument("--img-size", type = int, default = 224, help = "The input image size.")
	parser.add_argument("-nw", "--num-workers", type = int, default = 4, help = "The number of workers.")
	parser.add_argument("-d", "--device", type = str, default = "cuda", help = "The GPU device name.")
	return parser

def get_r3d_parser(parser):
	parser.add_argument("--pretrained-path", type = str, default = "./pretrained/r3d18_KM_200ep.pth", help = "The pretrained file path.")
	parser.add_argument("--model-depth", type = int, default = 18, help = "The depth of the resnet model.")
	parser.add_argument("--n-pretrained-classes", type = int, default = 1039, help = "The number of classes on the pretrained model.")
	return parser

def get_hisenc_parser(parser):
	parser.add_argument("--dropout", type = float, default = 0.5, help = "The dropout rate for AtLoc.")
	parser.add_argument("--feature-dim", type = int, default = 2048, help = "The feature dim for AtLoc.")
	parser.add_argument("--pretrained-path", type = str, default = "./pretrained/r3d18_KM_200ep.pth", help = "The pretrained file path.")
	return parser

def get_args(*reqs):
	parser = argparse.ArgumentParser()
	parser = get_base_parser(parser)
	for req in reqs:
		if req == "r3d":
			parser = get_r3d_parser(parser)
		elif req == "hisenc":
			parser = get_hisenc_parser(parser)
	return parser.parse_args()
