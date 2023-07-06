from torchvision import transforms

def get_img_transform(args):
	if args.img_size != -1:
		res = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
	else:
		res = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
	return res