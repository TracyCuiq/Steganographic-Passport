#pip install 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import math
from torch.cuda import amp
from math import exp
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms as T
#from natsort import natsorted
import glob
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import torchvision.models as models
#%matplotlib inline
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import logging
from datetime import datetime
from tqdm import tqdm
#from toolbox import *
import piqa
import argparse
from natsort import natsorted

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):

        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=2, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        else:
            self.split_len1 = in_1
            self.split_len2 = in_2
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1
        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)
        return torch.cat((y1, y2), 1)


class Hinet(nn.Module):

    def __init__(self):
        super(Hinet, self).__init__()

        self.inv1 = INV_block()
        self.inv2 = INV_block()
        self.inv3 = INV_block()
        self.inv4 = INV_block()
        
        self.inv5 = INV_block()
        self.inv6 = INV_block()
        self.inv7 = INV_block()
        self.inv8 = INV_block()

    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)

        else:
            out = self.inv8(x, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)
dwt = DWT()
iwt = IWT()

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def guide_loss(output, bicubic_image):
    #loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

def reconstruction_loss(rev_input, input):
    #loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(rev_input, input)
    return loss.to(device)

def low_frequency_loss(ll_input, gt_input):
    #loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = 0.01 * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).type(torch.uint8) / 255.
        return output.type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)
quantize = Quantization()

def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train", train_path='', val_path=''):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.files = natsorted(sorted(glob.glob(train_path + "/*." + 'png')))
        elif mode == 'se':
            self.files = sorted(glob.glob(val_path + "/*." + 'png'))
        else:
            # test
            self.files = sorted(glob.glob(val_path + "/*." + 'png'))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files)


def save_image(tensor, filename, nrow=4, padding=2,
            normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                    normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename, quality=100)

class AverageMeter_visual():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val 
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum

class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum

def evaluate_visual(my_inn, loader, token_image_ts):

    psnr_com = piqa.PSNR().cuda()
    ssim_com = piqa.SSIM().cuda()
    psnr_gen, psnr_res, psnr_rez = AverageMeter_visual(), AverageMeter_visual(), AverageMeter_visual()
    g_loss_sum, r_loss_sum, z_loss_sum = AverageMeter(), AverageMeter(), AverageMeter()
    my_inn.eval()
    for data in loader:
        data = data.cuda()
        with torch.no_grad():
            cover = data[data.shape[0]//2:, :, :, :]
            secret = data[:data.shape[0]//2, :, :, :]
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)
            #################
            #    forward:   #
            #################
            output = my_inn(input_img)
            output_steg = output.narrow(1, 0, 4 * 3)
            steg = iwt(output_steg)
            output_z = output.narrow(1, 4 * 3, output.shape[1] - 4 * 3)
            #output_z = gauss_noise(output_z.shape)
            output_z_iwt = iwt(output_z)
            #################
            #   backward:   #
            #################
            output_steg = output_steg.cuda()
            output_z_guass = dwt(token_image_ts)
            output_rev = torch.cat((output_steg, output_z_guass), 1)
            output_image = my_inn(output_rev, rev=True)
            secret_rev = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
            secret_rev = iwt(secret_rev)

            g_loss = guide_loss(steg, cover)
            r_loss = reconstruction_loss(secret_rev, secret) #+ reconstruction_loss(cover_rev, images)
            z_loss = reconstruction_loss(output_z_iwt, token_image_ts)

            cover_ = cover.clamp(min=0, max=1)
            steg_ = steg.clamp(min=0, max=1)
            output_z_iwt_ = output_z_iwt.clamp(min=0, max=1)
            token_image_ts_ = token_image_ts.clamp(min=0, max=1)
            secret_rev_ = secret_rev.clamp(min=0, max=1)
            secret_ = secret.clamp(min=0, max=1)
            psnr_g = psnr_com(cover_, steg_)
            psnr_z = psnr_com(output_z_iwt_, token_image_ts_)
            psnr_s = psnr_com(secret_rev_, secret_)

        psnr_gen.update(psnr_g, 1)
        psnr_res.update(psnr_s, 1)
        psnr_rez.update(psnr_z, 1)
        g_loss_sum.update(g_loss, len(cover_))
        r_loss_sum.update(r_loss, len(cover_))
        z_loss_sum.update(z_loss, len(cover_))

    return psnr_gen.average(), psnr_res.average(), psnr_rez.average(), g_loss_sum.average(), r_loss_sum.average(), z_loss_sum.average()

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default=224, help='image size')
    parser.add_argument('--train-bz', default=20, help='batch size for trianing')
    parser.add_argument('--test-bz', default=2, help='batch size for testing')
    parser.add_argument('--data-root', default='dataset', help='training and testing dataset root')
    parser.add_argument('--token-root', default='data/token/grooved_0072.jpg', help='root for the steganographic image')
    parser.add_argument('--checkpoint-root', default='checkpoint/', help='root for saving the checkpoint')
    parser.add_argument('--pretrained-inn', default='checkpoint/model_hinet.pt', help='pretrained inn root')
    parser.add_argument('--pretrained-inn-layer', default=8, help='the layers need to load of the pretrained inn model')
    parser.add_argument('--lr-inn', type=float, default=10 ** (-5.5), help='learning rate')
    parser.add_argument('--epochs', default=500, help='total training epochs')
    parser.add_argument('--save-interval', default=10, help='save checkpoints interval')
    parser.add_argument('--w-g', default=1, help='weighing para for gloss')
    parser.add_argument('--w-r', default=5, help='weighing para for rloss')
    parser.add_argument('--w-l', default=1, help='weighing para for lloss')
    args = parser.parse_args()

    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop(args.size),
        T.ToTensor()
    ])

    transform_val = T.Compose([
        T.CenterCrop(args.size),
        T.ToTensor(),
    ])

    ########################################
    batch_size_tr = args.train_bz
    batch_size_ts = args.test_bz
    # Training data loader
    clean_train_loader = DataLoader(
        Hinet_Dataset(transforms_=transform, mode="train", train_path=os.path.join(args.data_root, 'DIV2K_train_HR')),
        batch_size=batch_size_tr,#c.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True
    )
    # Test data loader
    clean_test_loader = DataLoader(
        Hinet_Dataset(transforms_=transform_val, mode="val", val_path=os.path.join(args.data_root, 'DIV2K_valid_HR')),
        batch_size=batch_size_ts,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )
    seloader = DataLoader(
        Hinet_Dataset(transforms_=transform_val, mode="se"),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=True
    )


    trans_transform = [
        #transforms.Resize([128, 128]),
        transforms.Resize([args.size, args.size]),
        #transforms.RandomResizedCrop([100, 100]),
        #transforms.Resize([128, 128]),
        transforms.ToTensor(),
    ]

    trans_transform = transforms.Compose(trans_transform)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    my_inn = Hinet().cuda()
    init_model(my_inn)
    #attack_list = ['Crop', 'Resize', 'Gaussian', 'Identity']
    #hidden_noise = Noiser(device=device, noise_layers=attack_list)
    scaler = amp.GradScaler()#.cuda()
    if args.pretrained_inn != None:
        PARAMS_PATH = args.pretrained_inn
        checkpoint = torch.load(PARAMS_PATH)
        #my_inn.load_state_dict({k.replace('module.model.', ''): v for k, v in checkpoint['net'].items()})        
        checkpoint_ = dict_slice(checkpoint['net'], 0, 30*args.pretrained_inn_layer)
        my_inn.load_state_dict({k.replace('module.model.', ''): v for k, v in checkpoint_.items()})
    
    params_trainable = (list(filter(lambda p: p.requires_grad, my_inn.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=args.lr_inn, betas=(0.5, 0.999), eps=1e-6, weight_decay=1e-5)

    condition = True
    train_idx = 0

    psnr_com = piqa.PSNR()
    ssim_com = piqa.SSIM().cuda()

    
    token_Path = args.token_root
    token_image = Image.open(token_Path)#.convert("RGB")
    token_image = trans_transform(token_image)
    token_image = token_image.cuda()
    token_image_tr = token_image.repeat(batch_size_tr//2, 1, 1, 1)
    token_image_ts = token_image.repeat(batch_size_ts//2, 1, 1, 1)

    idx_epoch = 0
    for idx_epoch in range(args.epochs):
        print('epoch', idx_epoch)
        
        # Train INN
        my_inn.train()
        for i, data in tqdm(enumerate(clean_train_loader), total=len(clean_train_loader)):

            data = data.to(device)
            cover = data[data.shape[0] // 2:]
            secret = data[:data.shape[0] // 2]
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            #print(cover_input.shape, secret_input.shape)
            input_img = torch.cat((cover_input, secret_input), 1)
            #################
            #    forward:   #
            #################
            output = my_inn(input_img)
            output_steg = output.narrow(1, 0, 4 * 3)
            output_z = output.narrow(1, 4 * 3, output.shape[1] - 4 * 3)
            steg_img = iwt(output_steg)
            #################
            #   backward:   #
            #################
            output_z_guass = dwt(token_image_tr)#gauss_noise(output_z.shape)
            output_rev = torch.cat((output_steg, output_z_guass), 1)
            output_image = my_inn(output_rev, rev=True)
            secret_rev = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
            secret_rev = iwt(secret_rev)
            #################
            #     loss:     #
            #################
            g_loss = guide_loss(steg_img, cover)
            r_loss = reconstruction_loss(secret_rev, secret)
            steg_low = output_steg.narrow(1, 0, 3)
            cover_low = cover_input.narrow(1, 0, 3)
            l_loss = low_frequency_loss(steg_low, cover_low)

            total_loss = args.w_r * r_loss + args.w_g * g_loss + args.w_l * l_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()

        # eval train set
        if idx_epoch % args.save_interval == 0:
            psnr_g_avg, psnr_s_avg, psnr_z_avg, g_loss, r_loss, z_loss = evaluate_visual(my_inn, clean_test_loader, token_image_ts)
            
            print('g_loss', g_loss.item(), 'r_loss', r_loss.item(), 'z_loss', z_loss.item())
            print(' PSNR', 'gen', psnr_g_avg.item(), 'res', psnr_s_avg.item(), 'rez', psnr_z_avg.item())
            #print(' PSNR', psnr_g, psnr_s)
            filename = os.path.join(args.checkpoint_root, 'my_inn_ep{}.pt'.format(idx_epoch))
            pickle.dump(my_inn, open(filename, 'wb'))

        filename = os.path.join(args.checkpoint_root, 'final.pt')
        pickle.dump(my_inn, open(filename, 'wb'))

if __name__ == '__main__':
    main()