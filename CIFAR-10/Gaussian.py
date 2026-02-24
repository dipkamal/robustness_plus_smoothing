
import os
import shutil
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import kornia
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from metrics import *

eps = 0.01
attack_steps = 7 
step_size = 0.0025
random_start= True

batch_size = 128
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
layers = 28
widen_factor = 10
droprate = 0
nesterov = True
max_num_training_steps = 70001
mile1 = 40000 
mile2 = 50000
eps = 0.0314
attack_steps = 7 
step_size = 0.00784
random_start= True

device = "cuda" if torch.cuda.is_available() else "cpu"


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)



class filter_denoising_block(torch.nn.Module):
    ''' Simple filters as denoising block'''
    def __init__(self, in_planes, ksize, filter_type):
        super(filter_denoising_block, self).__init__()
        self.in_planes = in_planes
        self.ksize = ksize
        self.filter_type = filter_type
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.filter_type == 'Median':
            x_denoised = kornia.filters.median_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Mean':
            x_denoised = kornia.filters.box_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Gaussian':
            x_denoised = kornia.filters.gaussian_blur2d(x, (self.ksize, self.ksize), (0.3 * ((x.shape[3] - 1) * 0.5 - 1) + 0.8, 0.3 * ((x.shape[2] - 1) * 0.5 - 1) + 0.8))
        new_x = x + self.conv(x_denoised)
        return new_x


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)

        # Denoising block
        self.denoise_block = filter_denoising_block(nChannels[1], ksize=3, filter_type='Gaussian')

        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)

        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)

        # Apply the denoising block
        out = self.denoise_block(out)

        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)



normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

train_loader = torch.utils.data.DataLoader(
        datasets.__dict__['CIFAR10']('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(
        datasets.__dict__['CIFAR10']('../data', train=False, transform=transform_test),
        batch_size=32, shuffle=True)

def evaluate_model(model, data, device):
    """Evaluate torch model."""
    model.eval()
    logits = torch.Tensor().to(device)
    targets = torch.LongTensor().to(device)

    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            logits = torch.cat([logits, model(images)])
            targets = torch.cat([targets, labels])

    predictions, labels = torch.nn.functional.softmax(logits, dim=1), targets
    test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())
    return test_acc


print('Gaussian denoise first layer')

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    """Train for one epoch on the training set"""

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate_model(model, data, device):
    """Evaluate torch model."""
    model.eval()
    logits = torch.Tensor().to(device)
    targets = torch.LongTensor().to(device)

    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            logits = torch.cat([logits, model(images)])
            targets = torch.cat([targets, labels])

    predictions, labels = torch.nn.functional.softmax(logits, dim=1), targets
    test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())
    return test_acc


#adv training

model = WideResNet(layers, 10 , widen_factor, dropRate=droprate)
model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum, nesterov = nesterov,
                                weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[mile1, mile2], gamma=0.1)

print('Gaussian denoise with adv first layer')


for step in range(0, max_num_training_steps):

    model.train()
    input, target = next(iter(train_loader))
    target = target.cuda(non_blocking=True)
    input = input.cuda(non_blocking=True)
    images = projected_gradient_descent(model, input, eps, step_size, attack_steps, np.inf)
    optimizer.zero_grad()

    # compute output
    output = model(images)
    loss = criterion(output, target)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step%500==0:
        print('step ', step)
        print(evaluate_model(model, val_loader, device))


path = 'Layer_1_Eps_D_Gaussian_advCIFARWideResNet.pth'
torch.save(model.state_dict(), path)


gc.collect()
torch.cuda.empty_cache()

del model
