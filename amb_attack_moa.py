import json
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import prepare_dataset
from experiments.utils import construct_passport_kwargs_from_dict
from models.alexnet_passport_private_moa import AlexNetPassportPrivate#TODO:
from models.resnet_passport_private_moa import ResNetPrivate
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private_moa import PassportPrivateBlock
from models.losses.sign_loss import SignLoss
import shutil
import wandb

class DatasetArgs():
    pass

def train_maximize(origpassport, fakepassport, model, optimizer, criterion, trainloader, device, type):
    model.train()
    loss_meter = 0
    signloss_meter = 0
    balloss_meter = 0
    maximizeloss_meter = 0
    mseloss_meter = 0
    csloss_meter = 0
    acc_meter = 0
    signacc_meter = 0
    start_time = time.time()
    mse_criterion = nn.MSELoss()
    cs_criterion = nn.CosineSimilarity()
    for k, (d, t) in enumerate(trainloader):
        d = d.to(device)
        t = t.to(device)

        loss = torch.tensor(0.).to(device)
        signloss = torch.tensor(0.).to(device)
        balloss = torch.tensor(0.).to(device)
        signacc = torch.tensor(0.).to(device)
        count = 0
        count_ = 0

        # optimizer.zero_grad()
        # pred = model(d, ind=1)  #fore branch
        # loss = criterion(pred, t)
        # for m in model.modules():
        #     if isinstance(m, SignLoss):
        #         signloss += m.loss
        #         signacc += m.acc
        #         count += 1

        for ind in range(2):
            pred = model(d, ind=ind)
            loss += criterion(pred, t)

            if ind == 1:
                # sign loss
                for m in model.modules():
                    if isinstance(m, SignLoss):
                        signloss += m.loss
                        signacc += m.acc
                        count += 1

                for m in model.modules():
                    if isinstance(m, PassportPrivateBlock):
                        balloss += m.get_loss()
                        count_ += 1

        if 'fake2-' in type: #TODO:
            #(loss + signloss + balloss).backward() 
            (loss).backward()  #only cross-entropy loss  backward  fake2

        elif  'fake3-' in type :
            maximizeloss = torch.tensor(0.).to(device)
            mseloss = torch.tensor(0.).to(device)
            csloss = torch.tensor(0.).to(device)
            for l, r in zip(origpassport, fakepassport):
                mse = mse_criterion(l, r)
                cs = cs_criterion(l.view(1, -1), r.view(1, -1)).mean()
                csloss += cs
                mseloss += mse
                maximizeloss += 1 / mse
            (loss + maximizeloss).backward()  #csloss do not backward   kafe3

        else:
            # print("FFFFFFFFFFFFFFFFFF")
            (loss + maximizeloss + signloss).backward()  #csloss  backward   #fake3_S

        torch.nn.utils.clip_grad_norm_(fakepassport, 2)  #梯度裁剪
        optimizer.step()
        acc = (pred.max(dim=1)[1] == t).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        signloss_meter += signloss.item()
        balloss_meter += balloss.item()
        signacc_meter += signacc.item() / count
        maximizeloss_meter += maximizeloss.item()
        mseloss_meter += mseloss.item()
        csloss_meter += csloss.item()

        print(f'Batch [{k + 1}/{len(trainloader)}]: '
              f'Loss: {loss_meter / (k + 1):.2f} '
              f'Acc: {100*acc_meter / (k + 1):.2f} '
              f'Bal Loss: {balloss_meter / (k + 1):.2f} '
              f'Sign Loss: {signloss_meter / (k + 1):.2f} '
              f'Sign Acc: {100*signacc_meter / (k + 1):.2f} '
              f'MSE Loss: {mseloss_meter / (k + 1):.2f} '
              f'Maximize Dist: {maximizeloss_meter / (k + 1):.2f} '
              f'CS: {csloss_meter / (k + 1):.2f} ({time.time() - start_time:.2f}s)',
              end='\r')

    print()
    loss_meter /= len(trainloader)
    acc_meter /= len(trainloader)
    signloss_meter /= len(trainloader)
    balloss_meter /= len(trainloader)
    signacc_meter /= len(trainloader)
    maximizeloss_meter /= len(trainloader)
    mseloss_meter /= len(trainloader)
    csloss_meter /= len(trainloader)

    return {'loss': loss_meter,
            'signloss': signloss_meter,
            'balloss': balloss_meter,
            'acc': acc_meter,
            'signacc': signacc_meter,
            'maximizeloss': maximizeloss_meter,
            'mseloss': mseloss_meter,
            'csloss': csloss_meter,
            'time': start_time - time.time()}




def train_ERB(model, optimizer, criterion, trainloader, device, type, ep):
    model.train()
    loss_meter = 0
    signloss_meter = 0
    balloss_meter = 0
    maximizeloss_meter = 0
    mseloss_meter = 0
    csloss_meter = 0
    dep_acc_meter = 0
    fore_acc_meter = 0
    signacc_meter = 0
    start_time = time.time()
    mse_criterion = nn.MSELoss()
    cs_criterion = nn.CosineSimilarity()
    for k, (d, t) in enumerate(trainloader):
        d = d.to(device)
        t = t.to(device)
        for m in model.modules():
            if isinstance(m, SignLoss):
                m.reset()

        loss = torch.tensor(0.).to(device)
        signloss = torch.tensor(0.).to(device)
        balloss = torch.tensor(0.).to(device)
        signacc = torch.tensor(0.).to(device)
        count, count_ = 0, 0

        pred = model(d, ind=0)
        loss += criterion(pred, t)
        acc = (pred.max(dim=1)[1] == t).float().mean()
        dep_acc_meter += acc.item()

        pred = model(d, ind=1)
        loss += criterion(pred, t)
        acc = (pred.max(dim=1)[1] == t).float().mean()
        fore_acc_meter += acc.item()

        # sign loss
        for m in model.modules():
            if isinstance(m, SignLoss):
                signloss += m.loss
                signacc += m.acc
                count += 1

        for m in model.modules():
            if isinstance(m, PassportPrivateBlock):
                balloss += m.get_loss()
                count_ += 1

        (loss + signloss ).backward()
        #(loss + signloss).backward()
        optimizer.step()

        loss_meter += loss.item()
        signloss_meter += signloss.item()
        #balloss_meter += balloss.item() / count_
        signacc_meter += signacc.item() / count

        print(f'Batch [{k + 1}/{len(trainloader)}]: '
              f'Loss: {loss_meter / (k + 1):.2f} '
              f'Fore. Acc: {100*fore_acc_meter / (k + 1):.2f} '
              f'Dep. Acc: {100*dep_acc_meter / (k + 1):.2f} '
              #f'Bal Loss: {balloss_meter / (k + 1):.2f} '
              f'Bal Dis: { 100*(np.abs(dep_acc_meter-fore_acc_meter)) / (k + 1):.2f} '
              f'Sign Loss: {signloss_meter / (k + 1):.2f} '
              f'Sign Acc: {100*signacc_meter / (k + 1):.2f} '
              ,
              end='\r')
        # if ep == 1:
        #     wandb.log({
        #             "AMB_ERB_training/Training loss": loss_meter / (k + 1) if loss_meter / (k + 1)<=200 else np.nan,
        #             "AMB_ERB_training/Sign Loss": signloss_meter / (k + 1) if signloss_meter / (k + 1)<=200 else np.nan,
        #             "AMB_ERB_training/Dep. acc": 100*dep_acc_meter / (k + 1),
        #             "AMB_ERB_training/Fore. acc": 100*fore_acc_meter / (k + 1),
        #             })

    print()
    loss_meter /= len(trainloader)
    fore_acc_meter /= len(trainloader)
    dep_acc_meter /= len(trainloader)
    signloss_meter /= len(trainloader)
    #balloss_meter /= len(trainloader)
    signacc_meter /= len(trainloader)


    return {'loss': loss_meter,
            'signloss': signloss_meter,
            #'balloss': balloss_meter,
            'depacc': dep_acc_meter,
            'foreacc': fore_acc_meter,
            'signacc': signacc_meter,
            'baldis': np.abs(dep_acc_meter-fore_acc_meter),
            'time': start_time - time.time()}



def test_fake(model, criterion, valloader, device):
    model.eval()
    loss_fore_meter, loss_dep_meter = 0, 0
    signloss_meter = 0
    acc_fore_meter, acc_dep_meter = 0, 0
    signacc_meter = 0
    #bal_dis_meter = 0
    start_time = time.time()

    with torch.no_grad():
        for k, (d, t) in enumerate(valloader):
            d = d.to(device)
            t = t.to(device)

            # if scheme == 1:
            #     pred = model(d)
            # else:
            for m in model.modules():
                if isinstance(m, SignLoss):
                    m.reset()

            pred_dep = model(d, ind=0)
            pred_fore = model(d, ind=1)
            loss_dep = criterion(pred_dep, t)
            loss_fore = criterion(pred_fore, t)
            
            signloss = torch.tensor(0.).to(device)
            #balloss = torch.tensor(0.).to(device)
            signacc = torch.tensor(0.).to(device)
            count, count_ = 0, 0

            # for m in model.modules():
            #     if isinstance(m, PassportPrivateBlock):
            #         # scale_, scale = m.get_scale(ind=1)
            #         # sl, s = m.get_scale_relu()
            #         # print('sl', sl.view(-1))
            #         balloss += m.get_loss() #NOTE: balance loss
            #         # print('scale4los', scale_.view(-1).detach().cpu().numpy())
            #         # print('scale', (scale).view(-1).detach().cpu().numpy() )
            #         count_ += 1


            for m in model.modules():
                if isinstance(m, SignLoss):
                    #signloss += m.get_loss()
                    signacc += m.get_acc()
                    count += 1

            acc_dep = (pred_dep.max(dim=1)[1] == t).float().mean()
            acc_fore = (pred_fore.max(dim=1)[1] == t).float().mean()

            loss_dep_meter += loss_dep.item()
            loss_fore_meter += loss_fore.item()

            acc_dep_meter += acc_dep.item()
            acc_fore_meter += acc_fore.item()
            #signloss_meter += signloss.item()
            #bal_dis_meter += balloss.item()
            try:
                signacc_meter += signacc.item() / count
            except:
                print('there is no sign loss module')
                raise ValueError


            print(f'Batch [{k + 1}/{len(valloader)}]: '
                  f'Fore. Loss: {loss_fore_meter / (k + 1):.4f} '
                  #f'Dep. Loss: {loss_dep_meter / (k + 1):.4f} '
                  f'Fore. Acc: {100*acc_fore_meter / (k + 1):.4f} '
                  f'Dep. Acc: {100*acc_dep_meter / (k + 1):.4f} '
                  #f'Bal. Loss: {bal_dis_meter / (k + 1):.4f} '
                  #f'Sign Loss: {signloss_meter / (k + 1):.4f} '
                  f'Sign Acc: {signacc_meter / (k + 1):.4f} '
                  f'Bal dis: {100*np.abs(acc_dep_meter-acc_fore_meter) / (k + 1):.4f} ({time.time() - start_time:.2f}s)',
                  end='\r')

    print()

    loss_fore_meter /= len(valloader)
    #loss_dep_meter /= len(valloader)
    acc_fore_meter /= len(valloader)
    acc_dep_meter /= len(valloader)
    #signloss_meter /= len(valloader)
    signacc_meter /= len(valloader)
    #bal_dis_meter /= len(valloader)

    return {'foreloss': loss_fore_meter,
            #'deploss': loss_dep_meter,
            #'balloss': bal_dis_meter,
            #'signloss': signloss_meter,
            'foreacc': acc_fore_meter,
            'depacc': acc_dep_meter,
            'signacc': signacc_meter,
            'baldis': np.abs(acc_dep_meter-acc_fore_meter),
            'time': time.time() - start_time}


