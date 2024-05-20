
import json
import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from dataset import prepare_dataset
from experiments.utils import construct_passport_kwargs_from_dict

from models.alexnet_passport_private_moa import AlexNetPassportPrivate
from models.resnet_passport_private_moa import ResNetPrivate
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private_moa import PassportPrivateBlock

import  shutil
from experiments.logger import  Logger, savefig
import matplotlib.pyplot as plt

def test(model, device, dataloader, msg='Testing Result', ind=0):
    model.eval()
    device = device
    verbose = True
    loss_meter = 0
    acc_meter = 0
    runcount = 0

    start_time = time.time()
    with torch.no_grad(): 
        for load in dataloader:
            data, target = load[:2]
            data = data.to(device)
            target = target.to(device)

            pred = model(data, ind=ind)
            loss_meter += F.cross_entropy(pred, target, reduction='sum').item()  # sum up batch loss
            #pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #acc_meter += pred.eq(target.view_as(pred)).sum().item()
            acc_meter += (pred.max(dim=1)[1] == target).float().sum()

            runcount += data.size(0)

    loss_meter /= runcount
    acc_meter = 100 * acc_meter / runcount

    if verbose:
        print(f'{msg}: '
              f'Loss: {loss_meter:6.4f} '
              f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
        print()

    return {'loss': loss_meter, 'acc': acc_meter, 'time': time.time() - start_time}

def test_signature(model):
    model.eval()
    res = {}
    avg_private = 0
    avg_public = 0
    count_private = 0
    count_public = 0

    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, PassportPrivateBlock):
                signbit, _ = m.get_scale_relu()
                signbit = signbit.view(-1).sign()
                privatebit = m.b

                detection = (signbit == privatebit).float().mean().item()
                res['private_' + name] = detection
                avg_private += detection
                count_private += 1

            # if isinstance(m, PassportBlock):
            #     signbit = m.get_scale().view(-1).sign()
            #     publicbit = m.b

            #     detection = (signbit == publicbit).float().mean().item()
            #     res['public_' + name] = detection
            #     avg_public += detection
            #     count_public += 1

    pub_acc = 0
    pri_acc = 0
    if count_private != 0:
        print(f'Private Sign Detection Accuracy: {avg_private / count_private * 100:6.4f}')
        pri_acc = avg_private / count_private

    # if count_public != 0:
    #     print(f'Public Sign Detection Accuracy: {avg_public / count_public * 100:6.4f}')


    return res,  pri_acc





def pruning_resnet(model, pruning_perc):
    if pruning_perc == 0:
        return

    allweights = []
    for p in model.parameters():
        allweights += p.data.cpu().abs().numpy().flatten().tolist()

    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)

    for name, p in model.named_parameters():
        if  'fc' not in name :
            mask = p.abs() > threshold
            p.data.mul_(mask.float())




def pruning_resnet2(model, pruning_perc, passport_config, type_prune='rd'):
    def prune_layer_random(layer, amount=0.3):
        prune.random_unstructured(module=layer, name="weight", amount=amount)

    def prune_layer_l1(layer, amount=0.4):
        prune.l1_unstructured(module=layer, name="weight", amount=amount)

    if pruning_perc == 0:
        return
    for name, module in model.named_modules():
        # Skip the layers you don't want to prune
        if isinstance(module, torch.nn.Conv2d):
            if type_prune == 'rd':
                prune_layer_random(module, pruning_perc/100)  # or prune_layer_l1(module)
            else:
                prune_layer_l1(module, pruning_perc/100)
        #elif isinstance(module, torch.nn.BatchNorm2d):
        #    prune_layer_random(module, pruning_perc/100)  # or prune_layer_l1(module)
        elif isinstance(module, torch.nn.Linear):
            if type_prune == 'rd':
                prune_layer_random(module, pruning_perc/100)  # or prune_layer_l1(module)
            else:
                prune_layer_l1(module, pruning_perc/100)

