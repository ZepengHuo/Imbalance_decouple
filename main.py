import argparse
import os
import torch
import random 
import numpy as np 
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
from itertools import cycle
import pickle

from resnetv2 import PreActResNet18 as ResNet18  
from utils import Labeled_dataset


parser = argparse.ArgumentParser(description='PyTorch Cifar10_100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--data_dir', help='The directory for data', default='trans_data', type=str)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='60,80', help='decreasing strategy')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='cifar10_cil', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')

best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()
    print(args)

    path_head = args.data_dir
    train_path = os.path.join(path_head,'4500_labeled_images_cifar10_train.pkl')
    val_path = os.path.join(path_head,'500_labeled_images_cifar10_val.pkl')
    old_img_path = os.path.join(path_head,'100_labeled_images_cifar10_train.pkl')
    test_path = os.path.join(path_head,'labeled_images_cifar10_test.pkl')
    sequence = np.random.permutation(10)
    print('class sequence: ', sequence)

    torch.cuda.set_device(int(args.gpu))

    if args.seed:
        setup_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    train_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor()
        ])

    val_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    model = ResNet18(num_classes=10)
    model.cuda()

    #prepare dataset
    train_dataset = Labeled_dataset(train_path, train_trans, sequence[8:10], offset=8)
    val_dataset = Labeled_dataset(val_path, val_trans, sequence[:10], offset=0)
    train_old_dataset = Labeled_dataset(old_img_path, train_trans, sequence[:8], offset=0)
    train_random_dataset = torch.utils.data.dataset.ConcatDataset((train_dataset,train_old_dataset))

    train_loader_random = torch.utils.data.DataLoader(
        train_random_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    train_loader_balance_new = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(args.batch_size/5), shuffle=True,
        num_workers=2, pin_memory=True)

    train_loader_balance_old = torch.utils.data.DataLoader(
        train_old_dataset,
        batch_size=int(args.batch_size*4/5), shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)


    train_acc = []
    ta=[]
    for epoch in range(args.epochs):
        print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))

        train_accuracy = train(train_loader_random, train_loader_balance_new, train_loader_balance_old, model, criterion, optimizer, epoch)

        prec1 = validate(val_loader, model, criterion, if_main=True)
        prec1 = validate(val_loader, model, criterion, if_main=False)

        train_acc.append(train_accuracy)
        ta.append(prec1)

        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer,
            }, filename=os.path.join(args.save_dir, 'best_model.pt'))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer,
        }, filename=os.path.join(args.save_dir, 'checkpoint.pt'))

        plt.plot(train_acc, label='train_acc')
        plt.plot(ta, label='TA')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()


def validate(val_loader, model, criterion, if_main=False):

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.long().cuda()

        # compute output
        with torch.no_grad():
            output = model(input, main_fc=if_main)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def train(rand_loader, new_balance_loader, old_balance_loader, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    coef_old = int(args.batch_size*4/5)/64
    coef_new = int(args.batch_size/5)/64

    # switch to train mode
    model.train()

    new_balance = iter(new_balance_loader)
    old_balance = iter(old_balance_loader)

    for i, (input, target) in enumerate(rand_loader):

        try:
            bal_new_img, bal_new_target = next(new_balance)
        except StopIteration:
            new_balance = iter(new_balance_loader)
            bal_new_img, bal_new_target = next(new_balance)

        try:
            bal_old_img, bal_old_target = next(old_balance)
        except StopIteration:
            old_balance = iter(old_balance_loader)
            bal_old_img, bal_old_target = next(old_balance)

        bal_new_img = bal_new_img.cuda()
        bal_old_img = bal_old_img.cuda()
        input = input.cuda()

        bal_new_target = bal_new_target.long().cuda()
        bal_old_target = bal_old_target.long().cuda()
        target = target.long().cuda()

        # random input
        output_gt = model(input, main_fc=False)
        loss_rand = criterion(output_gt, target)

        # balance inputs
        output_bal_new = model(bal_new_img, main_fc=True)
        output_bal_old = model(bal_old_img, main_fc=True)
        loss_balance = criterion(output_bal_new, bal_new_target)*coef_new + criterion(output_bal_old, bal_old_target)*coef_old

        loss = (loss_balance + loss_rand)*0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_gt.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(rand_loader), loss=losses, top1=top1))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

if __name__ == '__main__':
    main()