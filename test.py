import torch
from utils import *
from resnetmodel.res_mine import *
import argparse
import torch.nn as nn

if __name__ == '__main__':

    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='path to the stored model')
    parser.add_argument('--net_type', default='resnet', type=str, help='Type of net: resnet, alexnet, resnet_refined')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='Number of workers for dataloading')

    args = parser.parse_args()


    if args.net_type == 'Resnet':
        model = ResNet_mine(Bottleneck, [12, 12, 12], num_classes=100)
        model.load_state_dict(torch.load(args.src))
    else:
        model = ResNet18_mine()
        model.load_state_dict(torch.load(args.src))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)


    train_data, test_data,_,_ = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                            std=[0.2675, 0.2565, 0.2761], 
                                            dataset='CIFAR100',
                                            batch_size=256,
                                            num_workers=args.workers)


    criterion = nn.CrossEntropyLoss()

    temp_loss, temp_correct = test(test_data, model, criterion, 1)