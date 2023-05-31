import numpy as np
from sched import scheduler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode
import torch.nn.parallel

def dataloading(mean, std, dataset="CIFAR100", batch_size=128, num_workers=16, aug=True):
    normalize = transforms.Normalize(mean=mean, std=std)
    if aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
      

    if dataset=="CIFAR100":
        trainset = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
        testset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    else:
        trainset = datasets.CIFAR10('cifar10', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10('cifar10', train=False, download=True, transform=transform_test)

    train_data = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_data, test_data,trainset,testset

def pre_train(net,optimizer,criterion,batch_size,epochs,trainset,testloader,prob=0, beta=10, aug_type="baseline",scheduler=None):
    total_loss = []
    test_acc = []
    valid_acc = []
    l = len(trainset)
    train_size = int(0.8*l)
    valid_size = l-train_size
    for epoch in range(epochs):  # loop over the dataset multiple times

        train_, valid_ = torch.utils.data.random_split(trainset, [train_size, valid_size])
        trainloader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(valid_, batch_size=batch_size, shuffle=True, num_workers=2)
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # inputs, labels_a, labels_b = Variable(inputs), Variable(labels_a), Variable(labels_b)
            if prob == 0 or aug_type == "baseline":         
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            else:
                r = np.random.rand(1)
                if r < prob:
                    if aug_type == "cutmix":
                        """
                        Using cutmix augmentation
                        """
                        lam = np.random.beta(beta, beta)
                        rand_index = torch.randperm(inputs.size()[0]).cuda()
                        target_a = labels
                        target_b = labels[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (inputs.size()[-1] * inputs.size()[-2])

                        outputs = net(inputs)
                        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                    elif aug_type == "cutout":
                        """
                        Using cutout augmentation
                        """ 
                        lam = np.random.beta(beta, beta)
                        bbx1, bby1, bbx2, bby2 = rand_bbox(X.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = 0.0

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                    elif aug_type == "mixup":
                        """
                        Using mixup augmentation
                        """
                        lam = np.random.beta(beta, beta)
                        rand_index = torch.randperm(inputs.size()[0]).cuda()
                        target_a = labels
                        target_b = labels[rand_index]
                        inputs = inputs * lam + inputs[rand_index] * (1. - lam)

                        outputs = net(inputs)
                        loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                else:
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
            loss.backward() # 反向传播，计算参数的更新值

            optimizer.step() # 将计算得到的参数加到Net上
            
            epoch_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
                
        print('[%d, %5d] loss: %.3f' %(epoch + 1, epochs, epoch_loss))
        # for name, parms in model.named_parameters():
        #     print('-->name:',name)
        #     print('-->grad_value:',parms.grad)       
        
        total_loss.append(epoch_loss)
        # test
        acc_test = acc_on_test(net,testloader)
        acc_valid = acc_on_test(net,validloader)
        test_acc.append(acc_test)
        valid_acc.append(acc_valid)
    loss_per_epoch = [a / l for a in total_loss]
    print('Finished Training')
    return loss_per_epoch, valid_acc, test_acc

    
def train(train_data, net, criterion, optimizer, epoch, prob=0, beta=20, aug_type="cutmix", verbose=True):
    temp_loss = AverageMeter()
    temp_correct = AverageMeter()
    net.train()
    for inputs, labels in train_data:
        inputs = inputs.cuda()
        labels = labels.cuda()
        if prob == 0 or aug_type == "baseline":
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        else:
            r = np.random.rand(1)
            if r < prob:
                if aug_type == "cutmix":
                    """
                    Using cutmix augmentation
                    """
                    lam = np.random.beta(beta, beta)
                    rand_index = torch.randperm(inputs.size()[0]).cuda()
                    target_a = labels
                    target_b = labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (inputs.size()[-1] * inputs.size()[-2])

                    outputs = net(inputs)
                    loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
                elif aug_type == "cutout":
                    """
                    Using cutout augmentation
                    """ 
                    lam = np.random.beta(beta, beta)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = 0.0

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                elif aug_type == "mixup":
                    """
                    Using mixup augmentation
                    """
                    lam = np.random.beta(beta, beta)
                    rand_index = torch.randperm(inputs.size()[0]).cuda()
                    target_a = labels
                    target_b = labels[rand_index]
                    inputs = inputs * lam + inputs[rand_index] * (1. - lam)

                    outputs = net(inputs)
                    loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct,  = accuracy(outputs.data, labels)
        temp_correct.update(correct.item(), inputs.shape[0])
        temp_loss.update(loss.item(), inputs.shape[0])
        loss.detach()
    if verbose:
        print('epoch: %d, train loss: %.3f, train accuracy: %.2f' % (epoch, temp_loss.avg, temp_correct.avg))
    return temp_loss.avg, temp_correct.avg
     
def test(test_data, net, criterion, epoch, verbose=True):
    
    net.eval()
    temp_loss = AverageMeter()
    temp_correct = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            correct, = accuracy(outputs.data, labels)
            temp_correct.update(correct.item(), inputs.shape[0])
            #此时误差已经很小，因此用总loss
            #loss = loss / len(test_data.dataset)
            temp_loss.update(loss.item(), inputs.shape[0]) 
    if verbose:
        print('epoch: %d, test loss: %f, test accuracy: %f' % (epoch, temp_loss.avg, temp_correct.avg))
    return temp_loss.avg, temp_correct.avg

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def acc_on_test(model,testloader):
    correct = 0
    total = 0
    for input, target in testloader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        #torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        numbers, predic = torch.max(output,1)
        total += target.size(0) # 加一个batch_size
        correct += (predic==target).sum().item()
    acc = correct / total
    return acc

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    accu = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        accu.append(correct_k.mul_(1 / batch_size))

    return accu

