import wandb
import argparse
import shutil
import wandb
from datetime import datetime
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models.INN import Hinet, init_model, dwt, iwt, guide_loss, reconstruction_loss, INV_block, ResidualDenseBlock_out, Hinet_Dataset, gauss_noise
import torch
import piqa
import pickle
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from models.resnet_passport_private_moa import ResNet18Private
from experiments.utils import construct_passport_kwargs
import json
from dataset import prepare_test_CIFAR10
import torch.nn as nn
from amb_attack_moa import test_fake
import torch.nn.functional as F
import json

def clip_hw_to_even(tensor):
    # 获取tensor的高度和宽度
    H, W = tensor.shape[-2], tensor.shape[-1]
    
    # 计算偶数的高度和宽度
    even_H = H if H % 2 == 0 else H - 1
    even_W = W if W % 2 == 0 else W - 1
    
    # 使用切片来裁剪tensor
    return tensor[..., :even_H, :even_W]

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


class CustomKeyDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = []
        for sub_dir in os.listdir(main_dir):
            full_sub_dir = os.path.join(main_dir, sub_dir)
            if os.path.isdir(full_sub_dir):
                self.total_imgs.extend([os.path.join(full_sub_dir, img) for img in os.listdir(full_sub_dir) if img.endswith('.jpg')])

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image
    

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


class INNExperiment(object):

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_num = 100
        self.size = 256
        
        self.lr = args['lr_inn']
        self.token_Path = args['token_path']
        self.fake_token_Path = args['fake_token_path']
        self.bz = args['batch_size']
        if args['key_type'] == 'real':
            self.token_image = self.load_token_img()
            self.fake_token_image = self.load_fake_token_img()

        self.inn = Hinet().to(self.device)
        init_model(self.inn)
        self.pretrained_inn_path = args['pretrained_inn_path']
        self.val_path = args['val_path']
        self.passport_path = args['passport_path']

        
        self.logdir = args['log_path']
        if not args['train_inn']:
            self.logdir += ('key_type'+args['key_type']+'passport_type'+args['passport_type'])
        os.makedirs(self.logdir, exist_ok=True)
        self.save_img = args['save_img']
        self.save_img_path = args['save_img_path']
        self.eval_data = self.prepare_eval_dataset(path=self.val_path, bz=self.bz)
        #self.eval_passport_data = self.prepare_eval_dataset(path=self.passport_path, bz=self.bz)
        self.fake_key_data = self.prepare_eval_dataset(path=self.val_path, bz=self.bz, rd=77)
        self.fake_secret_path = args['fake_secret_path']
        #self.fake_key_path = args['fake_key_path']
        if self.save_img:
            self.save_cover_path = os.path.join(self.save_img_path, 'cover')
            self.save_stego_path = os.path.join(self.save_img_path, 'stego')
            self.save_secret_path = os.path.join(self.save_img_path, 'secret')
            self.save_resecret_path = os.path.join(self.save_img_path, 'resecret')
            self.save_fake_resecret_path = os.path.join(self.save_img_path, 'fake_resecret')
            self.save_fake_key_path = os.path.join(self.save_img_path, 'fake_key')
            os.makedirs(self.save_cover_path, exist_ok=True)
            os.makedirs(self.save_stego_path, exist_ok=True)
            os.makedirs(self.save_secret_path, exist_ok=True)
            os.makedirs(self.save_resecret_path, exist_ok=True)
            os.makedirs(self.save_fake_resecret_path, exist_ok=True)
            os.makedirs(self.save_fake_key_path, exist_ok=True)


    def prepare_eval_dataset(self, bz, path, rd=7):

        transform = transforms.Compose([
            transforms.CenterCrop([self.size, self.size]),
            transforms.ToTensor(),
        ])
        coco_dataset = CustomImageDataset(
            root_dir=path,  
            transform=transform
        )
        random.seed(rd)  # 设置随机种子以确保重复性
        selected_indices = random.sample(range(len(coco_dataset)), self.eval_num)
        subset_dataset = Subset(coco_dataset, selected_indices)
        dataloader = DataLoader(subset_dataset, batch_size=bz, shuffle=False)

        return dataloader

    def prepare_eval_att_dataset(self, bz, path, shuffle=False):

        transform = transforms.Compose([
            transforms.CenterCrop([self.size, self.size]),
            transforms.ToTensor(),
        ])
        loader = DataLoader(
            Hinet_Dataset(transforms_=transform, train_path=path),
            batch_size=bz,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=0,
            drop_last=True
        )
        return loader

    def prepare_forged_key_dataset(self, bz, path='', rd=47):

        transform = transforms.Compose([
            transforms.CenterCrop([self.size, self.size]),
            transforms.ToTensor(),
        ])
        dataset = CustomKeyDataset(
            main_dir=path,  
            transform=transform
        )
        random.seed(rd)  
        selected_indices = random.sample(range(len(dataset)), self.eval_num)
        subset_dataset = Subset(dataset, selected_indices)
        dataloader = DataLoader(subset_dataset, batch_size=bz, shuffle=False)
        return dataloader

    def load_token_img(self):

        trans_transform = [
            #transforms.Resize([128, 128]),
            transforms.Resize([self.size, self.size]),
            #transforms.RandomResizedCrop([100, 100]),
            #transforms.Resize([128, 128]),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        trans_transform = transforms.Compose(trans_transform)

        token_image = Image.open(self.token_Path)#.convert("RGB")
        token_image = trans_transform(token_image)
        token_image = token_image.to(self.device)
        # token_image_tr = token_image.repeat(batch_size_tr//2, 1, 1, 1)
        # token_image_ts = token_image.repeat(batch_size_ts//2, 1, 1, 1)
        token_image = token_image.repeat(self.bz//2, 1, 1, 1)
        return token_image

    def load_fake_token_img(self):

        trans_transform = [
            #transforms.Resize([128, 128]),
            transforms.Resize([self.size, self.size]),
            #transforms.Resize([128, 128]),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        trans_transform = transforms.Compose(trans_transform)

        token_image = Image.open(self.fake_token_Path)#.convert("RGB")
        token_image = trans_transform(token_image)
        token_image = token_image.to(self.device)
        # token_image_tr = token_image.repeat(batch_size_tr//2, 1, 1, 1)
        # token_image_ts = token_image.repeat(batch_size_ts//2, 1, 1, 1)
        token_image = token_image.repeat(self.bz//2, 1, 1, 1)
        return token_image

    
    def evaluate_visual(self):

        log_file_g = os.path.join(self.logdir, 'log_g.txt')
        log_file_r = os.path.join(self.logdir, 'log_r.txt')
        log_file_r_fake = os.path.join(self.logdir, 'log_r_fake.txt')
        lf_g = open(log_file_g, 'a')
        lf_r = open(log_file_r, 'a')
        lf_r_fake = open(log_file_r_fake, 'a')

        psnr_com = piqa.PSNR().to(self.device)
        ssim_com = piqa.SSIM().to(self.device)
        psnr_gen, psnr_res, psnr_rez = AverageMeter_visual(), AverageMeter_visual(), AverageMeter_visual()
        psnr_res_fake = AverageMeter_visual()
        g_loss_sum, r_loss_sum, z_loss_sum = AverageMeter(), AverageMeter(), AverageMeter()
        if self.pretrained_inn_path.endswith(".pt"):
            self.inn = pickle.load(open(self.pretrained_inn_path, 'rb'))
        elif self.pretrained_inn_path.endswith(".pkl"):
            checkpoint = torch.load(self.pretrained_inn_path)
            self.inn.load_state_dict(checkpoint['model_state_dict'])
        elif self.pretrained_inn_path.endswith(".pth"):
            self.inn.load_state_dict(torch.load(self.pretrained_inn_path))

        self.inn.eval()
        #token_image_ts = self.token_image 
        token_image_ts = self.fake_token_image#
        count = 0
        for data in self.eval_data:
            #data = clip_hw_to_even(data.to(self.device))
            data = data.to(self.device)
            #fake_key = fake_key.to(self.device)
            with torch.no_grad():
                cover = data[data.shape[0]//2:, :, :, :]
                secret = data[:data.shape[0]//2, :, :, :]
                cover_input = dwt(cover)
                secret_input = dwt(secret)
                input_img = torch.cat((cover_input, secret_input), 1)
                #################
                #    forward:   #
                #################
                output = self.inn(input_img)
                output_steg = output.narrow(1, 0, 4 * 3)
                steg = iwt(output_steg)
                output_z = output.narrow(1, 4 * 3, output.shape[1] - 4 * 3)
                output_z_fake = gauss_noise(output_z.shape)
                output_z_iwt = iwt(output_z)
                #################
                #   backward:   #
                #################
                output_steg = output_steg.cuda()
                output_z_guass = dwt(token_image_ts)
                output_rev = torch.cat((output_steg, output_z_guass), 1)
                output_image = self.inn(output_rev, rev=True)
                secret_rev = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
                secret_rev = iwt(secret_rev)

                output_steg = output_steg.cuda()
                output_z_guass = output_z_fake
                #output_z_guass = dwt(fake_key[0])
                output_rev = torch.cat((output_steg, output_z_guass), 1)
                output_image = self.inn(output_rev, rev=True)
                secret_rev__ = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
                secret_rev_fake = iwt(secret_rev__)

                g_loss = guide_loss(steg, cover)
                r_loss = reconstruction_loss(secret_rev, secret) #+ reconstruction_loss(cover_rev, images)
                z_loss = reconstruction_loss(output_z_iwt, token_image_ts)

                cover_ = cover.clamp(min=0, max=1)
                steg_ = steg.clamp(min=0, max=1)
                output_z_iwt_ = output_z_iwt.clamp(min=0, max=1)
                token_image_ts_ = token_image_ts.clamp(min=0, max=1)
                secret_rev_ = secret_rev.clamp(min=0, max=1)
                secret_rev_fake_ = secret_rev_fake.clamp(min=0, max=1)
                secret_ = secret.clamp(min=0, max=1)

                psnr_g = psnr_com(cover_, steg_)
                psnr_z = psnr_com(output_z_iwt_, token_image_ts_)
                psnr_s = psnr_com(secret_rev_, secret_)
                psnr_s_fake = psnr_com(secret_rev_fake_, secret_)

            psnr_gen.update(psnr_g, 1)
            psnr_res.update(psnr_s, 1)
            psnr_rez.update(psnr_z, 1)
            psnr_res_fake.update(psnr_s_fake, 1)

            g_loss_sum.update(g_loss, len(cover_))
            r_loss_sum.update(r_loss, len(cover_))
            z_loss_sum.update(z_loss, len(cover_))
            # wandb.log({
            #     "psnr_gen": psnr_g, 
            #     "psnr_res": psnr_s, 
            #     "psnr_res_fake": psnr_res_fake, 
            #     })
            print(psnr_g.item(), file=lf_g)
            lf_g.flush()
            print(psnr_s.item(), file=lf_r)
            lf_r.flush()
            print(psnr_s_fake.item(), file=lf_r_fake)
            lf_r_fake.flush()
            if self.save_img:
                z_img = iwt(output_z_fake)
                save_image(cover[0], (self.save_cover_path+'/'+str(count)+'.png'))
                save_image(secret[0], (self.save_secret_path+'/'+str(count)+'.png'))
                save_image(steg[0], (self.save_stego_path+'/'+ str(count)+ '.png'))
                save_image(secret_rev[0], (self.save_resecret_path+'/'+str(count)+'.png'))
                save_image(secret_rev_fake[0], (self.save_fake_resecret_path+'/'+str(count)+'.png'))
                #save_image(z_img[0], (self.save_fake_key_path+'/'+str(count)+'.png'))
                save_image(torch.abs(secret_rev[0]-secret[0])*5, (self.save_fake_key_path+'/'+str(count)+'.png'))
            count += 1

        return psnr_gen.average(), psnr_res.average(), psnr_rez.average(), g_loss_sum.average(), r_loss_sum.average(), z_loss_sum.average()


    def hide_reveal(self):

        log_file_g = os.path.join(self.logdir, 'log_g.txt')
        log_file_r = os.path.join(self.logdir, 'log_r.txt')
        log_file_r_fake_noise = os.path.join(self.logdir, 'log_r_fake_noise.txt')
        log_file_r_fake_random = os.path.join(self.logdir, 'log_r_fake_random.txt')
        # lf_g = open(log_file_g, 'a')
        # lf_r = open(log_file_r, 'a')
        lf_r_fake_noise = open(log_file_r_fake_noise, 'a')
        lf_r_fake_random = open(log_file_r_fake_random, 'a')

        psnr_com = piqa.PSNR().to(self.device)
        ssim_com = piqa.SSIM().to(self.device)
        psnr_gen, psnr_res, psnr_rez = AverageMeter_visual(), AverageMeter_visual(), AverageMeter_visual()
        psnr_res_fake_noise, psnr_res_fake_random = AverageMeter_visual(), AverageMeter_visual()
        g_loss_sum, r_loss_sum, z_loss_sum = AverageMeter(), AverageMeter(), AverageMeter()
        if self.pretrained_inn_path.endswith(".pt"):
            self.inn = pickle.load(open(self.pretrained_inn_path, 'rb'))
        elif self.pretrained_inn_path.endswith(".pkl"):
            checkpoint = torch.load(self.pretrained_inn_path)
            self.inn.load_state_dict(checkpoint['model_state_dict'])
        elif self.pretrained_inn_path.endswith(".pth"):
            self.inn.load_state_dict(torch.load(self.pretrained_inn_path))

        self.inn.eval()
        #token_image_ts = self.token_image 
        fake_loader = self.prepare_eval_dataset(bz=1, path=self.fake_secret_path)
        fake_data_iter = iter(fake_loader)
        
        #token_image_ts = self.fake_token_image#
        count = 0
        for data in self.eval_data:
            #data = clip_hw_to_even(data.to(self.device))
            if count == 4: # Specifying passport
                data = data.to(self.device)
                for fake in fake_loader:

                    token_image_ts = fake.to(self.device)
                    #fake_key = fake_key.to(self.device)
                    with torch.no_grad():
                        cover = data[data.shape[0]//2:, :, :, :]
                        secret = data[:data.shape[0]//2, :, :, :]
                        cover_input = dwt(cover)
                        secret_input = dwt(secret)
                        input_img = torch.cat((cover_input, secret_input), 1)
                        #################
                        #    forward:   #
                        #################
                        output = self.inn(input_img)
                        output_steg = output.narrow(1, 0, 4 * 3)
                        steg = iwt(output_steg)
                        output_z = output.narrow(1, 4 * 3, output.shape[1] - 4 * 3)
                        output_z_fake = gauss_noise(output_z.shape)
                        output_z_iwt = iwt(output_z)
                        #################
                        #   backward:   #
                        #################
                        output_steg = output_steg.cuda()
                        output_z_guass = dwt(token_image_ts)
                        output_rev = torch.cat((output_steg, output_z_guass), 1)
                        output_image = self.inn(output_rev, rev=True)
                        secret_rev = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
                        secret_rev_random = iwt(secret_rev)

                        output_steg = output_steg.cuda()
                        output_z_guass = output_z_fake
                        #output_z_guass = dwt(fake_key[0])
                        output_rev = torch.cat((output_steg, output_z_guass), 1)
                        output_image = self.inn(output_rev, rev=True)
                        secret_rev__ = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
                        secret_rev_noise = iwt(secret_rev__)

                        # g_loss = guide_loss(steg, cover)
                        # r_loss = reconstruction_loss(secret_rev, secret) #+ reconstruction_loss(cover_rev, images)
                        # z_loss = reconstruction_loss(output_z_iwt, token_image_ts)

                        #cover_ = cover.clamp(min=0, max=1)
                        #steg_ = steg.clamp(min=0, max=1)
                        #output_z_iwt_ = output_z_iwt.clamp(min=0, max=1)
                        #token_image_ts_ = token_image_ts.clamp(min=0, max=1)
                        secret_rev_random_ = secret_rev_random.clamp(min=0, max=1)
                        secret_rev_noise_ = secret_rev_noise.clamp(min=0, max=1)
                        secret_ = secret.clamp(min=0, max=1)

                        #psnr_g = psnr_com(cover_, steg_)
                        #psnr_z = psnr_com(output_z_iwt_, token_image_ts_)
                        psnr_s_random = psnr_com(secret_rev_random_, secret_)
                        psnr_s_noise = psnr_com(secret_rev_noise_, secret_)

                    #psnr_gen.update(psnr_g, 1)
                    # psnr_res.update(psnr_s, 1)
                    # psnr_rez.update(psnr_z, 1)
                    psnr_res_fake_random.update(psnr_s_random, 1)
                    psnr_res_fake_noise.update(psnr_s_noise, 1)

                    # g_loss_sum.update(g_loss, len(cover_))
                    # r_loss_sum.update(r_loss, len(cover_))
                    # z_loss_sum.update(z_loss, len(cover_))

                    print(psnr_s_noise.item(), file=lf_r_fake_noise)
                    lf_r_fake_noise.flush()
                    print(psnr_s_random.item(), file=lf_r_fake_random)
                    lf_r_fake_random.flush()

                break
            count += 1
        
        return psnr_res_fake_random.average(), psnr_res_fake_noise.average()


    def att_amb_passport(self):
        self.passport_config = json.load(open('passport_configs/resnet18_passport_l3.json'))
        self.norm_type = 'bn'
        self.key_type = 'image'
        self.sl_ratio = None
        passport_kwargs = construct_passport_kwargs(self)
        model = ResNet18Private(num_classes=10, passport_kwargs=passport_kwargs)
        sd = torch.load('/home/ruohan/MOA_copy/log/resnet_cifar10_v2_all-our4/25/models/best.pth')
        model.load_state_dict(sd)
        model.eval()
        cifar10_loader = prepare_test_CIFAR10()
        criterion_model = nn.CrossEntropyLoss()
        

        log_file_g = os.path.join(self.logdir, 'log_amb_g.txt')
        log_file_s = os.path.join(self.logdir, 'log_amb_s.txt')
        #log_file_g = os.path.join(self.logdir, 'log_g.txt')
        lf_g = open(log_file_g, 'a')
        lf_s = open(log_file_s, 'a')

        psnr_com = piqa.PSNR().to(self.device)
        ssim_com = piqa.SSIM().to(self.device)
        psnr_gen, psnr_res, psnr_rez = AverageMeter_visual(), AverageMeter_visual(), AverageMeter_visual()
        g_loss_sum, r_loss_sum, z_loss_sum = AverageMeter(), AverageMeter(), AverageMeter()
        if self.pretrained_inn_path.endswith(".pt"):
            self.inn = pickle.load(open(self.pretrained_inn_path, 'rb'))
        elif self.pretrained_inn_path.endswith(".pkl"):
            checkpoint = torch.load(self.pretrained_inn_path)
            self.inn.load_state_dict(checkpoint['model_state_dict'])
        elif self.pretrained_inn_path.endswith(".pth"):
            self.inn.load_state_dict(torch.load(self.pretrained_inn_path))
        self.inn.eval()

        save_stego_path = os.path.join(self.save_img_path, 'stego')
        save_cover_path = os.path.join(self.save_img_path, 'cover')
        save_secret_path = os.path.join(self.save_img_path, 'secret')


        self.passport_user = self.prepare_eval_dataset(path=save_stego_path, bz=1)
        self.passport_pro = self.prepare_eval_dataset(path=save_cover_path, bz=1)
        self.passport_secret = self.prepare_eval_dataset(path=save_secret_path, bz=1)
        #self.fake_secret = self.prepare_eval_dataset(path=self.fake_secret_path, bz=1)
        self.fake_secret = self.prepare_eval_dataset(path=self.val_path, bz=1, rd=74)

        token_image_ts = self.token_image
        self.psnr_att_passport_gen_dict, self.psnr_att_passport_res_dict = {}, {}
        self.psnr_att_passport_gen_list, self.psnr_att_passport_res_list = [], []
        count = 0


        for pu, pp, ps, fs in zip(self.passport_user, self.passport_pro, self.passport_secret, self.fake_secret):
            pu, pp, ps, fs = pu.to(self.device), pp.to(self.device), ps.to(self.device), fs.to(self.device)
            pu.requires_grad=True
            optim = torch.optim.Adam([pu], lr=1e-3, betas=(0.5, 0.999), eps=1e-6, weight_decay=1e-5)
            psnr_att_passport_gen_list, psnr_att_passport_res_list = [], []
            #dep_acc_list, ver_acc_list = [], []
            for i in range(1000):
                optim.zero_grad()
                #################
                #   backward:   #
                #################
                output_z_guass = dwt(token_image_ts)
                output_steg = dwt(pu)
                output_rev = torch.cat((output_steg, output_z_guass), 1)
                output_image = self.inn(output_rev, rev=True)
                secret_rev = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
                secret_rev = iwt(secret_rev)

                r_loss = reconstruction_loss(secret_rev, fs)
                r_loss.backward()
                torch.nn.utils.clip_grad_norm_(pu, 2)
                optim.step()

                cover_ = pp.clamp(0, 1)
                steg_ = pu.clamp(0, 1)
                secret_rev_ = secret_rev.clamp(0, 1)
                secret_ = fs.clamp(0, 1)
                psnr_g = psnr_com(cover_, steg_)
                psnr_s = psnr_com(secret_rev_, secret_)
                psnr_att_passport_gen_list.append(psnr_g.item())
                psnr_att_passport_res_list.append(psnr_s.item())
                #valres = test_fake(model, criterion_model, cifar10_loader, self.device)
            
            
            print('psnr_g', psnr_g.item(), ' ', 'psnr_s', psnr_s.item())
            self.psnr_att_passport_gen_dict[count] = psnr_att_passport_gen_list
            self.psnr_att_passport_res_dict[count] = psnr_att_passport_res_list
            #self.psnr_att_passport_gen_list.append(psnr_g.item())

            # print(psnr_g.item(), file=lf_g)
            # lf_g.flush()
            # print(psnr_s.item(), file=lf_s)
            # lf_s.flush()
            #self.psnr_att_passport_gen_list.append(psnr_g.item())
            #self.psnr_att_passport_res_list.append(psnr_s.item())

            with open('/home/ruohan/MOA_copy/log/attacks/att_amb_passport_psnr_gen_dict.json', 'w') as json_file:
                json.dump(self.psnr_att_passport_gen_dict, json_file)
            with open('/home/ruohan/MOA_copy/log/attacks/att_amb_passport_psnr_res_dict.json', 'w') as json_file:
                json.dump(self.psnr_att_passport_res_dict, json_file)
            count += 1
            if count > 99:
                break


    def plot_fake_passport(self):
        g_dict = self.psnr_att_passport_gen_dict
        s_dict = self.psnr_att_passport_res_dict

        s_data = np.array(list(s_dict.values()))
        s_mean_list = np.mean(s_data, axis=0).tolist()
        s_std_list = np.std(s_data, axis=0).tolist()

        g_data = np.array(list(g_dict.values()))
        g_mean_list = np.mean(g_data, axis=0).tolist()
        g_std_list = np.std(g_data, axis=0).tolist()

        plt.rcParams['font.size'] = 16
        colors = ['#87CEBF', '#8A5BC7', '#E57439']
        linestyles = ['-', '--', '-.']
        fig = plt.figure()
        
        x = np.arange(len(s_mean_list))
        #x2 = np.arange(len(k_mean_list))

        #ax1.set_ylabel('Means', color='tab:blue')
        plt.plot(x, s_mean_list, color='#329C86',  label='Avg, secret')
        #ax1.tick_params(axis='y', labelcolor='tab:blue')
        plt.fill_between(x, np.subtract(s_mean_list, s_std_list), np.add(s_mean_list, s_std_list), color='#87CEBF', alpha=0.2, label='Std, secret')

        #ax2 = ax1.twinx()
        #ax2.set_ylabel('Meank', color='tab:red') 
        plt.plot(x, g_mean_list, color='#620BD3', label='Avg, passport')
        #ax2.tick_params(axis='y', labelcolor='tab:red')
        plt.fill_between(x, np.subtract(g_mean_list, g_std_list), np.add(g_mean_list, g_std_list), color='#8A5BC7', alpha=0.2, label='Std, passport')

        # 添加图例
        fig.legend(fontsize=10, framealpha=0.5)

        plt.grid(True)
        plt.ylabel('PSNR')
        plt.xlabel('Iteration')
        plt.xticks()
        plt.yticks()
        
        #plt.subplots_adjust(right=0.7)
        fig.tight_layout()

        save_name = self.logdir + '/fake_passport.pdf'
        plt.savefig(save_name, dpi=150)


    def att_amb_key(self):
        log_file_k = os.path.join(self.logdir, 'log_amb_key_k.txt') # real & fake key PSNR
        log_file_s = os.path.join(self.logdir, 'log_amb_key_s.txt') # extract PSNR

        lf_k = open(log_file_k, 'a')
        lf_s = open(log_file_s, 'a')

        psnr_com = piqa.PSNR().to(self.device)
        ssim_com = piqa.SSIM().to(self.device)
        psnr_gen, psnr_res, psnr_rez = AverageMeter_visual(), AverageMeter_visual(), AverageMeter_visual()
        g_loss_sum, r_loss_sum, z_loss_sum = AverageMeter(), AverageMeter(), AverageMeter()
        if self.pretrained_inn_path.endswith(".pt"):
            self.inn = pickle.load(open(self.pretrained_inn_path, 'rb'))
        elif self.pretrained_inn_path.endswith(".pkl"):
            checkpoint = torch.load(self.pretrained_inn_path)
            self.inn.load_state_dict(checkpoint['model_state_dict'])
        elif self.pretrained_inn_path.endswith(".pth"):
            self.inn.load_state_dict(torch.load(self.pretrained_inn_path))
        self.inn.eval()
        save_stego_path = os.path.join(self.save_img_path, 'stego')
        save_cover_path = os.path.join(self.save_img_path, 'cover')
        save_secret_path = os.path.join(self.save_img_path, 'secret')
        self.passport_user = self.prepare_eval_att_dataset(path=save_stego_path, bz=1)
        self.passport_owner = self.prepare_eval_att_dataset(path=save_cover_path, bz=1)
        self.genuine_user_ID = self.prepare_eval_att_dataset(path=save_secret_path, bz=1)
        self.forged_user_ID = self.prepare_eval_dataset(path=self.val_path, bz=1, rd=88)

        #self.fake_key = self.prepare_eval_dataset(path=self.fake_key_path, bz=1)

        genuine_key = self.token_image
        #fake_token_image

        self.psnr_att_key_s_dict, self.psnr_att_key_k_dict = {}, {}
        count = 0
        #for pu, pp, ps in zip(self.passport_user, self.passport_owner, self.forged_user_ID):
        for pu, pp, ps in zip(self.passport_user, self.passport_owner, self.genuine_user_ID):
            pu, pp, ps = pu.to(self.device), pp.to(self.device), ps.to(self.device)
            pu__ = dwt(pu)
            fk = torch.randn(pu__.shape).to(self.device)
            fk.requires_grad = True
            optim = torch.optim.Adam([fk], lr=1e-3, betas=(0.5, 0.999), eps=1e-6, weight_decay=1e-5)
            psnr_att_key_s_list, psnr_att_key_k_list = [], []
            for i in range(1000): # attemps
                optim.zero_grad()
                #################
                #   backward:   #
                #################
                #output_z_guass = dwt(token_image_ts)
                output_steg = dwt(pu)
                output_rev = torch.cat((output_steg, fk), 1)
                output_image = self.inn(output_rev, rev=True)
                secret_rev = output_image.narrow(1, 4 * 3, output_image.shape[1] - 4 * 3)
                secret_rev = iwt(secret_rev)

                r_loss = reconstruction_loss(secret_rev, ps)
                r_loss.backward()
                torch.nn.utils.clip_grad_norm_(pu, 2)
                optim.step()

                secret_rev_ = secret_rev.clamp(0, 1)
                secret_ = ps.clamp(0, 1)
                psnr_s = psnr_com(secret_rev_, secret_)
                fk_ = iwt(fk)
                fk_ = fk_.clamp(0, 1)
                psnr_k = psnr_com(genuine_key, fk_)
                psnr_att_key_s_list.append(psnr_s.item())
                psnr_att_key_k_list.append(psnr_k.item())
                
                wandb.log(
                    {'PSNR_S': psnr_s.item(),
                    'PSNR_K': psnr_k.item(),},
                    step = i
                )
            self.psnr_att_key_s_dict[count] = psnr_att_key_s_list
            self.psnr_att_key_k_dict[count] = psnr_att_key_k_list

            print('psnr_s', psnr_s.item(), ' ', 'psnr_k', psnr_k.item())
            count += 1

            with open('./log/attacks/att_amb_key_genuineID_psnr_att_key_s_dict.json', 'w') as json_file:
                json.dump(self.psnr_att_key_s_dict, json_file)
            with open('./log/attacks/att_amb_key_genuineID_psnr_att_key_k_dict.json', 'w') as json_file:
                json.dump(self.psnr_att_key_k_dict, json_file)
        

    def plot_fake_key(self):
        s_dict = self.psnr_att_key_s_dict
        k_dict = self.psnr_att_key_k_dict
        # print('s_dict', s_dict)
        # print('k_dict', k_dict)

        s_data = np.array(list(s_dict.values()))
        s_mean_list = np.mean(s_data, axis=0).tolist()
        s_std_list = np.std(s_data, axis=0).tolist()
        #s_std_list = np.std(s_data, ddof=1).tolist()
        # print('s_mean_list', s_mean_list)
        # print('s_std_list', s_std_list)


        k_data = np.array(list(k_dict.values()))
        k_mean_list = np.mean(k_data, axis=0).tolist()
        k_std_list = np.std(k_data, axis=0).tolist()
        # print('k_mean_list', k_mean_list)
        # print('k_std_list', k_std_list)


        plt.rcParams['font.size'] = 24
        colors = ['#87CEBF', '#8A5BC7', '#E57439']
        linestyles = ['-', '--', '-.']
        fig = plt.figure()
        x = np.arange(len(s_mean_list))
        #x2 = np.arange(len(k_mean_list))

        #ax1.set_ylabel('Means', color='tab:blue')
        plt.plot(x, s_mean_list, color='#329C86',  label='Avg, secret')
        #ax1.tick_params(axis='y', labelcolor='tab:blue')
        plt.fill_between(x, np.subtract(s_mean_list, s_std_list), np.add(s_mean_list, s_std_list), color='#87CEBF', alpha=0.2, label='Std, secret')

        #ax2 = ax1.twinx()
        #ax2.set_ylabel('Meank', color='tab:red') 
        plt.plot(x, k_mean_list, color='#620BD3', label='Avg, key')
        #ax2.tick_params(axis='y', labelcolor='tab:red')
        plt.fill_between(x, np.subtract(k_mean_list, k_std_list), np.add(k_mean_list, k_std_list), color='#8A5BC7', alpha=0.2, label='Std, key')

        # 添加图例
        fig.legend(fontsize=10, framealpha=0.5)

        plt.grid(True)
        plt.ylabel('PSNR')
        plt.xlabel('Iteration')
        plt.xticks()
        plt.yticks()
        
        #plt.subplots_adjust(right=0.7)
        fig.tight_layout()

        save_name = self.logdir + '/fake_key_coco.pdf'
        plt.savefig(save_name, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    # passport argument
    parser.add_argument('--batch-size', type=int, default=2, help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs (default: 200)')
    parser.add_argument('--lr-inn', type=float, default=10 ** (-5.5), help='learning rate (default: 0.01)')
    parser.add_argument('--train-inn', action='store_true', default=False,  help='joint train inn')
    parser.add_argument('--key-type', choices=['real', 'random', 'forge'], default='real', help='INN key type')
    parser.add_argument('--passport-type', choices=['userside', 'ownerside', 'forge'], default='userside', help='passport type')

    # paths
    parser.add_argument('--pretrained-inn-path', default='checkpoint/my_inn_token491.pth', help='load pretrained path')
    #parser.add_argument('--trigger-path', help='passport trigger path')
    parser.add_argument('--token-path', default='data/token/grooved_0072.jpg', help=' token path')
    parser.add_argument('--passport-path', default='data/trigger_set/pics', help=' token path')
    parser.add_argument('--fake-token-path', default='data/fake_token/000000006609.jpg', help=' token path')
    parser.add_argument('--val-path', default='dataset/COCO2017/test2017', help=' token path')
    parser.add_argument('--fake-secret-path', default='dataset/COCO2017/test2017', help=' token path')
    parser.add_argument('--log-path', default='/home/ruohan/MOA_copy/log/INN', help=' log path')
    parser.add_argument('--save-img-path', default='data/gen_COCO', help=' log path')

    # misc
    parser.add_argument('--save-img', action='store_true', help='for evaluation')
    parser.add_argument('--amb-att', action='store_true', help='for evaluation')
    parser.add_argument('--amb-att-key', action='store_true', help='for evaluation')
    
    args = parser.parse_args()
    exp = INNExperiment(vars(args))


    expname = 'MOA_INN'
    expname += args.key_type
    expname += args.passport_type

    current_datetime = datetime.now()
    # Convert to string in the format YYYY-MM-DD
    datetime_string = current_datetime.strftime('%Y-%m-%d %H:%M')
    date = str(datetime_string)
    
    wandb.init(
        project="MOA",
        name=expname,
        # track hyperparameters and run metadata
        config={
        #"learning_rate": exp.lr,
        #"dataset": exp.dataset,
        #"epochs": exp.epochs//1,
        "date": date,
        }
    )

    if not args.train_inn:
        if args.amb_att:
            if args.amb_att_key:
                exp.att_amb_key()
                #exp.plot_fake_key()
            else:
                exp.att_amb_passport()
                exp.plot_fake_passport()
        else:
            # psnr_gen,  psnr_res, psnr_rez, _, _, _ = exp.evaluate_visual()
            # print('Avg PSNR: ', psnr_gen, psnr_res)
            psnr_rd,  psnr_noise = exp.hide_reveal()
            print('Avg PSNR: ', psnr_rd, psnr_noise)
        


if __name__ == '__main__':
    main()
