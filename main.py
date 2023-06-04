import argparse
from ast import parse
from ctypes import resize
import os
import shutil
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils import *
import numpy as np
from resnetmodel.res_mine import *
import copy
import warnings

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Data augmentation of CIFAR100')
	parser.add_argument('--lr', default=0.1, type=float, metavar='LR')
	parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N')
	parser.add_argument('--epochs', default=100, type=int, metavar='N')
	parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',)
	parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W')
	parser.add_argument('--beta', default=10, type=float, help='hyperparameter beta')
	parser.add_argument('--dataset', default='CIFAR100', type=str)
	parser.add_argument('--net_type', default='Resnet', type=str)
	parser.add_argument('--workers', default=16, type=int, metavar='N')
	parser.add_argument('--depth', default=110, type=int,help='depth of the network of resnet type (default: 110)')
	parser.add_argument('--num_classes', default=100, type=int, metavar='N')
	parser.add_argument('-tpath', '--tensorboard_path', default='', type=str,help='tensorboard writer path, store in train and test')
	parser.add_argument('-tlabel', '--tensorboard_label', default='', type=str,help='tensorboard label, Loss and Acc')
	parser.add_argument('--verbose', default=True, type=bool,help='print the status at the end of every epoch')
	parser.add_argument('--aug', default=True, type=bool,help='Initial Data augmentation')
	parser.add_argument('--aug_prob', default=0.5, type=float,help='Data augmentation probability')
	parser.add_argument('--aug_type', default='cutmix', type=str,help='data augmentation type: cutmix, mixup, cutout')
	parser.add_argument('--model_save', default=True, type=bool,help='store the best model')
	parser.add_argument('--save_path', default='./models/cutmix.pth', type=str,help='path to store the best model')
	parser.add_argument('--bottleneck', default=True, type=bool,help='use bottleneck for resnet')
	parser.add_argument('--scheduler', default=True, type=bool,help='Use cosine scheduler for lr decay')
	parser.add_argument('--restart', default=2, type=int, metavar='N',help='number of restart phases')
	parser.add_argument('--mult', default=1, type=int, metavar='N', help="mult factor of number of epochs after each restart")


	args = parser.parse_args()

	train_writer = SummaryWriter(args.tensorboard_path + '/train')
	test_writer = SummaryWriter(args.tensorboard_path + '/test')


	train_data, test_data,_,_ = dataloading(mean=[0.4914, 0.4822, 0.4465], 
											std=[0.2675, 0.2565, 0.2761], 
											dataset=args.dataset,
											batch_size=args.batch_size, 
											num_workers=args.workers,
											aug=args.aug)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	if args.net_type == 'Resnet':
		if args.bottleneck:
			n = (args.depth - 2) // 9
			model = ResNet_mine(Bottleneck, [n, n, n], num_classes=args.num_classes)
		else:    
			n = (args.depth - 2) // 6
			model = ResNet_mine(BasicBlock, [n, n, n], num_classes=args.num_classes)
	elif args.net_type == 'ResNet18':
		model = ResNet18_mine(num_classes=100)

	#其他的网络暂时不考虑
	else:
		pass


	net = model.cuda()


	lr = 0.1
	criterion = nn.CrossEntropyLoss().cuda()
	best_acc = 0
	epoch_counter = 0
	for j in range(args.restart):
		print("===Start of %d restart phase===" % (j + 1))
		optimizer = optim.SGD(net.parameters(),
							momentum=args.momentum, 
							lr = args.lr, 
							weight_decay=args.weight_decay, 
							nesterov=True)
		if args.scheduler:
			scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (args.mult ** j))
		for i in range(args.epochs * (args.mult ** j)):
			if not args.scheduler:
				if i == args.epochs // 2 or i == args.epochs * 3 // 4:
					for param_group in optimizer.param_groups:
						param_group['lr'] = param_group['lr'] * 0.1
		
			for param_group in optimizer.param_groups:
				print(param_group['lr'])
		
			temp_loss, temp_correct = train(train_data, 
											net, 
											criterion, 
											optimizer, 
											epoch_counter + 1, 
											args.aug_prob, 
											args.beta, 
											aug_type=args.aug_type, 
											verbose=args.verbose)	

			train_writer.add_scalar(args.tensorboard_label + 'Loss', temp_loss, epoch_counter + 1)
			train_writer.add_scalar(args.tensorboard_label + 'Acc', temp_correct, epoch_counter + 1)

			
			temp_loss, temp_correct = test(test_data, 
											net, 
											criterion, 
											epoch_counter + 1,
											verbose=args.verbose)

			test_writer.add_scalar(args.tensorboard_label + 'Loss', temp_loss, epoch_counter + 1)
			test_writer.add_scalar(args.tensorboard_label + 'Acc', temp_correct, epoch_counter + 1)
			
			if best_acc < temp_correct:
				best_acc = temp_correct
				if args.model_save:
					model_cache = copy.deepcopy(net.state_dict())
		
			if args.scheduler:
				scheduler.step()	
			
			epoch_counter = epoch_counter + 1
		print("===End of %d restart phase===" % (j + 1))
	
	if args.model_save:
		torch.save(model_cache, args.save_path)
	print("Best accuracy on testing instances: %.2f" % best_acc)