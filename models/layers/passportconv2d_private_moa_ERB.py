import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from models.losses.sign_loss import SignLoss
from .hash import custom_hash


class PassportPrivateBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, passport_kwargs={}):
        super().__init__()

        if passport_kwargs == {}:
            print('Warning, passport_kwargs is empty')

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias=False) # NOTE: this is Wc
        self.relu = nn.ReLU(inplace=True)

        self.key_type = passport_kwargs.get('key_type', 'random')
        self.weight = self.conv.weight

        self.alpha = passport_kwargs.get('sign_loss', 1)
        self.norm_type = passport_kwargs.get('norm_type', 'bn')

        self.init_public_bit = passport_kwargs.get('init_public_bit', True)
        self.requires_reset_key = False
        
        self.register_buffer('key_private', None)
        self.register_buffer('skey_private', None)

        self.init_scale(True)
        self.init_bias(True)

        norm_type = passport_kwargs.get('norm_type', 'bn')

        if norm_type == 'bn':
            self.bn0 = nn.BatchNorm2d(o, affine=False)
            self.bn1 = nn.BatchNorm2d(o, affine=False)
        elif norm_type == 'nose_bn':
            self.bn0 = nn.BatchNorm2d(o, affine=False)
            self.bn1 = nn.BatchNorm2d(o, affine=False)
        elif norm_type == 'gn':
            self.bn0 = nn.GroupNorm(o // 16, o, affine=False)
            self.bn1 = nn.GroupNorm(o // 16, o, affine=False)
        elif norm_type == 'in':
            self.bn0 = nn.InstanceNorm2d(o, affine=False)
            self.bn1 = nn.InstanceNorm2d(o, affine=False)
        elif norm_type == 'sbn' or norm_type == 'sbn_se':
            self.bn = nn.BatchNorm2d(o, affine=False)
        else:
            self.bn = nn.Sequential()

        hid = 1 if o // 4 == 0 else o // 4
        self.fc = nn.Sequential(
            nn.Linear(o, hid, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hid, o, bias=True),
        )

        ########################
        # NOTE: set bit info  
        ########################
        b = passport_kwargs.get('b', torch.sign(torch.rand(o) - 0.5))  # bit information to store， second para is the default value

        if isinstance(b, int):
            b = torch.ones(o) * b

        if isinstance(b, str):
            if len(b) * 8 > o:
                raise Exception('Too much bit information')
            bsign = torch.sign(torch.rand(o) - 0.5)
            bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])

            for i, c in enumerate(bitstring):
                if c == '0':
                    bsign[i] = -1
                else:
                    bsign[i] = 1
            b = bsign

        self.register_buffer('b', b) # set self.b as the para buffer
        self.sign_loss_private = SignLoss(self.alpha, self.b) # NOTE: set self.b to init Signloss, correlated the forwoard fidelity
        self.l1_loss = nn.L1Loss()
        self.reset_parameters()


    def update_b(self, b):
        self.b = b
        self.sign_loss_private = SignLoss(self.alpha, b)

    def set_b(self, y):
        b = custom_hash(y, hash_length=len(self.b)) #NOTE: bit information to store by using Alice signitures
        b = b.to(self.b.device)
        self.b = b
        self.sign_loss_private = SignLoss(self.alpha, b)


    def init_bias(self, force_init=False):
        if force_init:
            self.bias = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device)) #relu bias
            self.bias0 = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device)) #bn0 bias
            self.bias1 = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device)) #bn1 bias
            self.bias_ERB = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.zeros_(self.bias)
            init.zeros_(self.bias0)
            init.zeros_(self.bias1)
            init.zeros_(self.bias_ERB)
        else:
            self.bias = None

    def init_scale(self, force_init=False):
        if force_init:
            self.scale = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device)) #relu scale
            self.scale0 = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device)) #bn0 scale
            self.scale1 = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device)) #bn1 scale
            self.scale_ERB = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.ones_(self.scale)
            init.ones_(self.scale0)
            init.ones_(self.scale1)
            init.ones_(self.scale_ERB)
        else:
            self.scale = None

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def passport_selection(self, passport_candidates):
        b, c, h, w = passport_candidates.size()

        if c == 3:  # input channel
            randb = random.randint(0, b - 1)
            return passport_candidates[randb].unsqueeze(0)

        passport_candidates = passport_candidates.view(b * c, h, w)
        full = False
        flag = [False for _ in range(b * c)]
        channel = c
        passportcount = 0
        bcount = 0
        passport = []

        while not full:
            if bcount >= b:
                bcount = 0

            randc = bcount * channel + random.randint(0, channel - 1)
            while flag[randc]:
                randc = bcount * channel + random.randint(0, channel - 1)
            flag[randc] = True

            passport.append(passport_candidates[randc].unsqueeze(0).unsqueeze(0))

            passportcount += 1
            bcount += 1

            if passportcount >= channel:
                full = True

        passport = torch.cat(passport, dim=1)
        return passport

    def set_key(self, x, y=None):
        n = int(x.size(0))
        if n != 1:
            x = self.passport_selection(x)
            if y is not None:
                y = self.passport_selection(y)
        

        #https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        # assert x.size(0) == 1, 'only batch size of 1 for key'
        self.register_buffer('key_private', x) #NOTE: this is bias
        # assert y is not None and y.size(0) == 1, 'only batch size of 1 for key'
        self.register_buffer('skey_private', y) #NOTE: this is scale
        self.set_b(y)
        


    def get_scale_key(self):
        return self.skey_private

    def get_bias_key(self):
        return self.key_private
    

    def get_scale_relu(self):
        skey = self.skey_private # get the scale key image (backdoor image) that stored in the buffuer
        scale_loss = self.sign_loss_private # get the sinloss object
        scalekey = self.conv(skey) # cal the cov feuture maps
        ##########################
        #NOTE: global average pooling
        ###########################
        b, c = scalekey.size(0), scalekey.size(1)
        kernel_size = scalekey.size()[2:]
        avg_pooled = F.avg_pool2d(scalekey, kernel_size)
        scale = avg_pooled.view(avg_pooled.size(0), avg_pooled.size(1))
        scale = scale.mean(dim=0) # global average pooling value

        scale_ = self.fc(scale).view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1) + scale_

        
        if scale_loss is not None:
            scale_loss.reset()
            scale_loss.add(scale_)

        return scale_, scale


    def get_scale_bn(self, force_passport=False, ind=0):
        if self.scale is not None and not force_passport:
            if ind == 0: #NOTE: this is the public branch
                return self.scale0.view(1, -1, 1, 1)
            elif ind == 1:
                return self.scale1.view(1, -1, 1, 1)
            else:
                raise ValueError
        

    def get_bias_relu(self):
        key = self.key_private # get the bias key image (backdoor image) that stored in the buffuer
        biaskey = self.conv(key)  # key batch always 1
        ##########################
        #NOTE: global average pooling
        ###########################
        b, c = biaskey.size(0), biaskey.size(1)
        bias = biaskey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        bias = bias.mean(dim=0).view(1, c, 1, 1)

        bias = bias.view(1, c)
        bias = self.fc(bias).view(1, c, 1, 1)
        return bias

    def get_bias_bn(self, force_passport=False, ind=0):
        if self.bias is not None and not force_passport:
            if ind == 0:
                return self.bias0.view(1, -1, 1, 1)
            elif ind == 1:
                return self.bias1.view(1, -1, 1, 1)
            else:
                raise ValueError
       
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        keyname = prefix + 'key_private'
        skeyname = prefix + 'skey_private'

        if keyname in state_dict:
            self.register_buffer('key_private', torch.randn(*state_dict[keyname].size()))
        if skeyname in state_dict:
            self.register_buffer('skey_private', torch.randn(*state_dict[skeyname].size()))

        scalename = prefix + 'scale'
        biasname = prefix + 'bias'
        if scalename in state_dict:
            self.scale = nn.Parameter(torch.randn(*state_dict[scalename].size()))

        if biasname in state_dict:
            self.bias = nn.Parameter(torch.randn(*state_dict[biasname].size()))

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


    def generate_key(self, *shape):
        newshape = list(shape)
        newshape[0] = 1

        min = -1.0
        max = 1.0
        key = np.random.uniform(min, max, newshape)
        print('random key generated')
        return key


    def get_loss(self): #NOTE: this is the balance loss
        _, scale = self.get_scale_relu() # [1, c, 1, 1]
        bias = self.get_bias_relu()
        scale = scale.view(-1)
        bias = bias.view(-1)
        loss = self.l1_loss(self.scale0, self.scale1 * scale) + self.l1_loss(self.bias0, self.bias1 * scale + bias)
        return loss
    

    def forward(self, x, force_passport=False, ind=0):

        key = self.key_private
        if (key is None and self.key_type == 'random') or self.requires_reset_key:
            self.set_key(torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device),
                         torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device))
        scale0 = self.scale0.view(1, -1, 1, 1)
        scale1 = self.scale1.view(1, -1, 1, 1)
        bias0 = self.bias0.view(1, -1, 1, 1)
        bias1 = self.bias1.view(1, -1, 1, 1)

        x = self.conv(x)

        if ind==0:
            x = self.bn0(x)
            x = scale0 * x + bias0
        else:
            x = self.bn1(x)
            x = scale1 * x + bias1
            _, scale = self.get_scale_relu()
            bias = self.get_bias_relu()
            x = scale * x + bias

        x = self.relu(x)

        return x
