from cgi import test
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import torchvision
import torchvision.transforms as transforms
import tqdm
from resnetmodel.res_mine import *
from tqdm import trange
from utils import *
import matplotlib.pyplot as plt
from utils import pre_train,acc_on_test

if __name__ == '__main__':
    #----------------------------------------------------------------------------------------#
    #--------------------------------------0、参数设置（已初步调参）-------------------------------------------#
    lr = 0.1
    model = ResNet18_mine(num_classes=100).cuda()
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80,120,150], gamma=0.1)
    epoch = 100
    batchsize = 512

    #----------------------------------------------------------------------------------------#
    #--------------------------------------1、baseline-------------------------------------------#

    train_data, test_data,trainset,testset = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                            std=[0.2675, 0.2565, 0.2761], 
                                            dataset='CIFAR100',
                                            batch_size=256,
                                            num_workers=2,aug=False)

    loss,val_acc, test_acc = pre_train(model,optimizer,criterion,batchsize, epoch,trainset,test_data,scheduler)
    data = {'loss':loss,'val_acc':val_acc,'test_acc':test_acc}
    torch.save(data,'./datas/baseline.pth')


    #----------------------------------------------------------------------------------------#
    #--------------------------------------2、baseline+augmentation-------------------------------------------#

    train_data, test_data,trainset,testset = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                            std=[0.2675, 0.2565, 0.2761], 
                                            dataset='CIFAR100',
                                            batch_size=256,
                                            num_workers=2,aug=True)

    loss,val_acc, test_acc = pre_train(model,optimizer,criterion,batchsize, epoch,trainset,test_data,scheduler)
    data = {'loss':loss,'val_acc':val_acc,'test_acc':test_acc}
    torch.save(data,'./datas/baseline_aug_data.pth')


    #----------------------------------------------------------------------------------------#
    #--------------------------------------3、baseline+mixup-------------------------------------------#

    train_data, test_data,trainset,testset = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                            std=[0.2675, 0.2565, 0.2761], 
                                            dataset='CIFAR100',
                                            batch_size=256,
                                            num_workers=2,aug=False)

    loss,val_acc, test_acc = pre_train(model,optimizer,criterion,batchsize,epoch,trainset,test_data,prob=0.5,aug_type="mixup",scheduler=scheduler)
    data = {'loss':loss,'val_acc':val_acc,'test_acc':test_acc}
    torch.save(data,'./datas/mixup_only.pth')


    #----------------------------------------------------------------------------------------#
    #--------------------------------------4、baseline+aug+mixup-------------------------------------------#

    train_data, test_data,trainset,testset = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                            std=[0.2675, 0.2565, 0.2761], 
                                            dataset='CIFAR100',
                                            batch_size=256,
                                            num_workers=2,aug=True)

    loss,val_acc, test_acc = pre_train(model,optimizer,criterion,batchsize,epoch,trainset,test_data,prob=0.5,aug_type="mixup",scheduler=scheduler)
    data = {'loss':loss,'val_acc':val_acc,'test_acc':test_acc}
    torch.save(data,'./datas/mixup_aug.pth')


    #----------------------------------------------------------------------------------------#
    #--------------------------------------5、baseline+cutmix-------------------------------------------#

    train_data, test_data,trainset,testset = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                            std=[0.2675, 0.2565, 0.2761], 
                                            dataset='CIFAR100',
                                            batch_size=256,
                                            num_workers=2,aug=False)

    loss,val_acc, test_acc = pre_train(model,optimizer,criterion,batchsize,epoch,trainset,test_data,prob=0.5,aug_type="cutmix",scheduler=scheduler)
    data = {'loss':loss,'val_acc':val_acc,'test_acc':test_acc}
    torch.save(data,'./datas/cutmix_only.pth')


    #----------------------------------------------------------------------------------------#
    #--------------------------------------6、baseline+aug+cutmix-------------------------------------------#

    train_data, test_data,trainset,testset = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                            std=[0.2675, 0.2565, 0.2761], 
                                            dataset='CIFAR100',
                                            batch_size=256,
                                            num_workers=2,aug=True)

    loss,val_acc, test_acc = pre_train(model,optimizer,criterion,batchsize,epoch,trainset,test_data,prob=0.5,aug_type="cutmix",scheduler=scheduler)
    data = {'loss':loss,'val_acc':val_acc,'test_acc':test_acc}
    torch.save(data,'./datas/cutmix_aug.pth')


    #----------------------------------------------------------------------------------------#
    #--------------------------------------7、baseline+cutout-------------------------------------------#
    train_data, test_data,trainset,testset = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                            std=[0.2675, 0.2565, 0.2761], 
                                            dataset='CIFAR100',
                                            batch_size=256,
                                            num_workers=2,aug=False)

    loss,val_acc, test_acc = pre_train(model,optimizer,criterion,batchsize,epoch,trainset,test_data,prob=0.5,aug_type="cutout",scheduler=scheduler)
    data = {'loss':loss,'val_acc':val_acc,'test_acc':test_acc}
    torch.save(data,'./datas/cutmix_only.pth')


    #----------------------------------------------------------------------------------------#
    #--------------------------------------8、baseline+aug+cutout-------------------------------------------#
    train_data, test_data,trainset,testset = dataloading(mean=[0.4914, 0.4822, 0.4465], 
                                            std=[0.2675, 0.2565, 0.2761], 
                                            dataset='CIFAR100',
                                            batch_size=256,
                                            num_workers=2,aug=True)

    loss,val_acc, test_acc = pre_train(model,optimizer,criterion,batchsize,epoch,trainset,test_data,prob=0.5,aug_type="cutout",scheduler=scheduler)
    data = {'loss':loss,'val_acc':val_acc,'test_acc':test_acc}
    torch.save(data,'./datas/cutmix_aug.pth')