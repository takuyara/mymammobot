import os
import argparse

from ds_gen.video_sequence import generate_video_sequence

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mesh-path", type = str, default = "./meshes/Airway_Phantom_AdjustSmooth.stl")
	parser.add_argument("--output-path", type = str, default = "./virtual_dataset/video_sequence")
	parser.add_argument("--cl-path", type = str, default = "./CL")
	parser.add_argument("--img-size", type = int, default = 224)
	parser.add_argument("--num-samples", type = int, default = 5)
	parser.add_argument("--partition", type = str, default = "train")
	#parser.add_argument("--out-pose-only", action = "store_true", default = False)
	return parser.parse_args()

def main():
	args = get_args()
	output_path = os.path.join(args.output_path, args.partition)
	os.makedirs(output_path, exist_ok = True)

	generate_video_sequence(args.mesh_path, args.cl_path, output_path, args.num_samples, args.img_size)

if __name__ == '__main__':
	main()
