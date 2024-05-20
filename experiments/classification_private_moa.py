import os
import sys
import random
import pickle
import wandb
import collections
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import make_interp_spline

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms
from torchvision.utils import save_image

import passport_generator
from dataset import *#prepare_dataset, prepare_wm, prepare_emb#, prepare_attck_dataset

from experiments.baseMOA import Experiment
from experiments.trainer import Trainer
from experiments.trainer_private_moa import TrainerPrivate, TesterPrivate
from experiments.utils import construct_passport_kwargs
from experiments.logger import Logger, savefig
from models.alexnet_normal import AlexNetNormal
from models.alexnet_passport_private_moa import AlexNetPassportPrivate
from models.layers.conv2d import ConvBlock
from models.resnet_normal import ResNet18
from models.resnet_passport_private_moa import ResNet18Private
from models.INN import *
from models.layers.hash import custom_hash
from models.layers.passportconv2d import PassportBlock 
from models.layers.passportconv2d_private_moa import PassportPrivateBlock

from prun_moa import test, test_signature, pruning_resnet2
from amb_attack_moa import train_maximize, test_fake, train_ERB


class ClassificationPrivateExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)

        self.in_channels = 1 if self.dataset == 'mnist' else 3
        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech-101': 101,
            'caltech-256': 256
        }[self.dataset]

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])
        #############
        #prepare data
        #############
        self.train_data, self.valid_data = prepare_dataset(self.args)
        self.wm_data = None

        if self.use_trigger_as_passport:
            self.passport_data = prepare_wm(self.trigger_path)
        else:
            self.passport_data = self.valid_data
        self.emb_data = prepare_emb(self.emb_path, num=2, is_tar=True)
        
        if self.train_backdoor:
            self.wm_data = prepare_wm(self.trigger_path)

        #############
        #construct model
        #############
        self.construct_model()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        
        if len(self.lr_config[self.lr_config['type']]) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self.lr_config[self.lr_config['type']], self.lr_config['gamma'])
        else:
            scheduler = None
        self.trainer = TrainerPrivate(self.model, optimizer, scheduler, self.device)

        if self.is_tl: # transfer learnign
            self.wandb_expid = self.finetune_load()
        else:
            self.wandb_expid = self.makedirs_or_load()

    def construct_model(self):

        def setup_keys():
            if self.key_type != 'random':
                if self.arch == 'alexnet':
                    pretrained_model = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
                elif self.arch == 'resnet':
                    pretrained_model = ResNet18(num_classes=self.num_classes, norm_type=self.norm_type)
                else:
                    raise NotImplementedError
                pretrained_model = pretrained_model.to(self.device)
                self.setup_keys(pretrained_model)

        ########## passport_config setting##############################
        passport_kwargs = construct_passport_kwargs(self)
        self.passport_kwargs = passport_kwargs
        ################################################################

        if self.arch == 'alexnet':
            model = AlexNetPassportPrivate(self.in_channels, self.num_classes, passport_kwargs)
        elif self.arch == 'resnet':
            model = ResNet18Private(num_classes=self.num_classes, passport_kwargs=passport_kwargs)
        else:
            raise NotImplementedError
            
        self.model = model.to(self.device)
        # self.model_inn = Hinet().to(self.device)
        # self.pretrained_inn_path = self.args['pretrained_inn_path']
        # pickle.dump(self.model_inn, open(self.pretrained_inn_path, 'wb'))
        setup_keys()

    def setup_keys(self, pretrained_model):
        if self.key_type != 'random':
            n = 1 if self.key_type == 'image' else 20  # any number
            key_x, x_inds = passport_generator.get_key_sig(self.passport_data, n)
            key_y, y_inds = passport_generator.get_key_sig(self.passport_data, n)
            key_x, key_y = key_x.to(self.device), key_y.to(self.device)
            passport_generator.set_key(pretrained_model, self.model, key_x, key_y)

    def training(self):
        best_acc = float('-inf')
        history_file = os.path.join(self.logdir, 'history.csv')
        best_file = os.path.join(self.logdir, 'best.txt')
        f = open(best_file, 'a')
        best_ep = 1
        first = True
        
        if self.save_interval > 0:
            self.save_model('epoch-0.pth')

        for ep in range(1, self.epochs + 1):
            print(f'-----------epoch-{ep}--Training---------')
            train_metrics = self.trainer.train(ep, self.train_data, self.wm_data, self.arch)
            print(f'Sign Detection Train Accuracy: {train_metrics["sign_acc"] :.4f}')

            print('------------Testing-----------------------')
            valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')
            wandb.log({
                "train/Priv. acc": train_metrics["acc_private"], "train/Pub. acc": train_metrics["acc_public"], 
                "train/loss": train_metrics["loss"], "train/sign loss": train_metrics["sign_loss"], 
                "train/sign acc": train_metrics["sign_acc"], "train/balance loss": train_metrics["balance_loss"],
                "val/Priv. acc": valid_metrics["acc_private"], "val/Pub. acc": valid_metrics["acc_public"], "val/Priv. loss": valid_metrics["loss_private"], "val/Pub. loss": valid_metrics["loss_public"]
            })

            wm_metrics = {}
            if self.train_backdoor:
                print('--------Testing--WM-backdoor-----------')
                wm_metrics = self.trainer.test(self.wm_data, 'WM Result')
                wm_metrics = self.trainer.test_backdoor(self.wm_data, 'WM INN Result', self.emb_data)


            metrics = {}
            for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]
            for key in wm_metrics: metrics[f'wm_{key}'] = wm_metrics[key]
            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')

            if best_acc < metrics['valid_total_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_total_acc']
                self.save_model('best.pth')
                best_ep = ep

            self.save_last_model()

            f = open(best_file,'a')
            f.write(str(best_acc) + "\n")
            print(str(wm_metrics) + '\n', file=f)
            print(str(metrics) + '\n',file=f)
            f.write( "\n")
            f.write("best epoch: %s"%str(best_ep) + '\n')
            f.flush()

    def evaluate(self):
        self.trainer.test(self.valid_data)


    def transfer_learning(self):
        if not self.is_tl:
            raise Exception('Please run with --transfer-learning')

        if self.tl_dataset == 'caltech-101':
            self.num_classes = 101
        elif self.tl_dataset == 'cifar100':
            self.num_classes = 100
        elif self.tl_dataset == 'caltech-256':
            self.num_classes = 256
        elif self.tl_dataset == 'cifar10':  # 
            self.num_classes = 10
        else:
            raise ValueError

        print('Loading model')
        self.load_model('best.pth')
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)

        if len(self.lr_config[self.lr_config['type']]) != 0:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.lr_config[self.lr_config['type']],
                                                       self.lr_config['gamma'])
        else:
            scheduler = None
        scheduler = None
        tl_trainer = Trainer(self.model, optimizer, scheduler, self.device)
        valid_metrics = tl_trainer.test(self.valid_data)
        metrics = {}
        for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]
        print(valid_metrics)


    def pruning(self):
        device = self.device
        logdir = self.logdir
        load_path = os.path.split(logdir)[0] 
        #print('self.logdir', self.logdir)
        prun_dir = self.logdir + '/prun'
        if not os.path.exists(prun_dir):
            os.mkdir(prun_dir)

        title = ''
        txt_pth = os.path.join(prun_dir, 'log_prun.txt')
        logger_prun = Logger(txt_pth, title=title)
        #logger_prun.set_names(['Deployment', 'Verification', 'Signature', 'Diff'])
        logger_prun.set_names(['Deployment, Norm', 'Verification, Norm', 'Signature, Norm', 
                               'Deployment, Random', 'Verification, Random', 'Signature, Random'])

        txt_pth2 = os.path.join(prun_dir, 'log_prun2.txt')
        logger_prun2 = Logger(txt_pth2, title=title)
        logger_prun2.set_names(['Deployment', 'Verification', 'Signature'])

        self.train_data, self.valid_data = prepare_dataset(self.args)
        print('loadpath--------', load_path)

        for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

            sd = torch.load(load_path + '/models/best.pth')
            model_copy = copy.deepcopy(self.model)
            model_copy2 = copy.deepcopy(self.model)

            model_copy.load_state_dict(sd)
            model_copy2.load_state_dict(sd)
            model_copy.to(self.device)
            model_copy2.to(self.device)
            pruning_resnet2(model_copy, perc, self.passport_config, type_prune='l1')
            pruning_resnet2(model_copy2, perc, self.passport_config, type_prune='rd')
            
            res = {}
            res2 = {}
            res_wm = {}

            #self.wm_data = prepare_wm('data/trigger_set/pics')
            res['perc'] = perc
            res['pub_ori'] = test(model_copy, device, self.valid_data, msg='pruning %s percent Dep. Result' % perc, ind=0)
            res['pri_ori'] = test(model_copy, device, self.valid_data, msg='pruning %s percent Ver. Result' % perc, ind=1)
            _, res['pri_sign_acc'] = test_signature(model_copy)

            res2['perc'] = perc
            res2['pub_ori'] = test(model_copy2, device, self.valid_data, msg='pruning %s percent Dep. Result' % perc, ind=0)
            res2['pri_ori'] = test(model_copy2, device, self.valid_data, msg='pruning %s percent Ver. Result' % perc, ind=1)
            _, res2['pri_sign_acc'] = test_signature(model_copy2)
            #res_wm['pri_ori'] = test(self.model, device, self.wm_data, msg='pruning %s percent Pri_Trigger Result' % perc, ind=1)
            del model_copy, model_copy2

            pub_acc = res['pub_ori']['acc']
            pri_acc = res['pri_ori']['acc']
            #pri_acc_wm = res_wm['pri_ori']['acc']
            pri_sign_acc = res['pri_sign_acc'] * 100

            pub_acc2 = res2['pub_ori']['acc']
            pri_acc2 = res2['pri_ori']['acc']
            pri_sign_acc2 = res2['pri_sign_acc'] * 100

            diff = torch.abs(pub_acc-pri_acc)
            logger_prun.append([pub_acc, pri_acc, pri_sign_acc, pub_acc2, pri_acc2, pri_sign_acc2 ])
            #logger_prun.append([pub_acc, pri_acc, pri_sign_acc])
            #logger_prun2.append([pub_acc2, pri_acc2, pri_sign_acc2])

        
    def fake_attack(self):

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

        def prepare_attck_dataset(rd=42):

            data_dir = 'dataset/COCO2017'
            mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            transform = transforms.Compose([
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            coco_dataset = CustomImageDataset(
                root_dir=f'{data_dir}/test2017', 
                transform=transform
            )
            random.seed(rd)
            selected_indices = random.sample(range(len(coco_dataset)), 200)
            subset_dataset = Subset(coco_dataset, selected_indices)
            batch_size = 1
            dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

            return dataloader
        
        epochs = self.epochs
        lr = 0.01
        device = self.device
        
        loadpath = self.logdir
        print('loadpath: ', loadpath)
        task_name = loadpath.split('/')[-2]
        print('task_name', task_name)
        loadpath_all = loadpath + '/models/best.pth'
        sd = torch.load(loadpath_all)
        #shutil.copy('amb_attack_moa.py', str(logdir) + "/amb_attack_moa.py")
        ################################
        # NOTE: random attack
        ################################
        if self.pro_att == 'fix':
            logdir = 'log/attacks/' + task_name+ '/Random_ATT/MOA/' + str(self.wandb_expid) + '/'
            os.makedirs(logdir, exist_ok=True)
            best_file = os.path.join(logdir, 'best.txt')
            log_file = os.path.join(logdir, 'log.txt')
            lf = open(log_file, 'a')
            self.model.load_state_dict(sd, strict=False)
            self.model.to(self.device)
            print(loadpath_all + ' is loaded')

            for param in self.model.parameters():
                param.requires_grad_(False)

            passblocks = []
            origpassport = []
            fakepassport = []

            def run_cs():  
                cs = []
                for d1, d2 in zip(origpassport, fakepassport):
                    d1 = d1.view(d1.size(0), -1)
                    d2 = d2.view(d2.size(0), -1)
                    cs.append(F.cosine_similarity(d1, d2).item())
                return cs

            print('Running fix var provider-side attack!')
            att_data = prepare_attck_dataset()
            criterion = nn.CrossEntropyLoss()
            id_count = 0
            if self.arch == 'alexnet':
                tl_model = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
            elif self.arch == 'resnet':
                tl_model = ResNet18(num_classes=self.num_classes, norm_type=self.norm_type)
            else:
                raise NotImplementedError
            tl_model = tl_model.to(device)
           
            for batch in att_data:
                receipt_img0, receipt_img1 = batch
                receipt_img0 = receipt_img0.unsqueeze(0).to(device)
                receipt_img1 = receipt_img1.unsqueeze(0).to(device)
                save_image(receipt_img0, './tmp/'+'output_image' + str(id) + '.png')

                for m in self.model.modules():
                    if isinstance(m, PassportPrivateBlock):
                        passblocks.append(m)

                        keyname = 'key_private'
                        skeyname = 'skey_private'
                        bname = 'b'
                        scale_fore_name = 'scale1'
                        bias_fore_name = 'bias1'

                        key, skey = m.__getattr__(keyname).data.clone(), m.__getattr__(skeyname).data.clone()
                        origpassport.append(key.to(device))
                        origpassport.append(skey.to(device))
                        scale1, bias1 = m.__getattr__(scale_fore_name).data.clone(), m.__getattr__(bias_fore_name).data.clone()

                        m.__delattr__(keyname) 
                        m.__delattr__(skeyname)
                        m.__delattr__(scale_fore_name)
                        m.__delattr__(bias_fore_name)

                        noise1 = torch.ones(*scale1.size()).to(device)
                        noise2 = torch.zeros(*bias1.size()).to(device) #* 100
                        m.register_buffer(scale_fore_name, nn.Parameter( noise1 ))
                        m.register_buffer(bias_fore_name, nn.Parameter( noise2 ))

                with torch.no_grad():
                    self.model.set_intermediate_keys(self.model, receipt_img0, receipt_img1)
                res = {}
                valres = test_fake(self.model, criterion, self.valid_data, device)
                for key in valres: res[f'valid_{key}'] = valres[key]

                print(res)
                print(valres["foreacc"] * 100, file=lf)
                lf.flush()
                # sys.exit(0)
                id_count += 1
                wandb.log({
                    "sign_acc": valres["signacc"] * 100, 
                    "Dep. acc": valres["depacc"] * 100, 
                    "Fore. acc": valres["foreacc"] * 100, 
                    "bal_dis": valres["depacc"]* 100 - valres["foreacc"]* 100, 
                    })
            return

        ################################
        # NOTE: ERB attack
        ################################
        elif self.pro_att == 'ERB': 
            with_TLP = True
            TLP_flag = 'TLP' if with_TLP else ''

            from models.resnet_passport_private_moa_ERB import ResNet18Private
            from models.alexnet_passport_private_moa_ERB import AlexNetPassportPrivate
            self.train_data, self.valid_data = prepare_ERB_dataset(self.args)
            lr = 1e-4

            logdir = 'log/attacks/' + task_name+ '/ERB_ATT/MOA'+TLP_flag +'/'+ str(self.wandb_expid) + '/'
            os.makedirs(logdir, exist_ok=True)
            best_file = os.path.join(logdir, 'best.txt')
            log_file = os.path.join(logdir, 'log.txt')
            lf = open(log_file, 'a')

            self.model.load_state_dict(sd, strict=True)
            self.model.to(self.device)
            print(loadpath_all + ' is loaded')
            att_data_x = prepare_attck_dataset(7)
            att_data_y = prepare_attck_dataset(71)
            origpassport = []
            fakepassport = []

            if self.arch == 'alexnet':
                sub_model = AlexNetPassportPrivate(self.in_channels, self.num_classes, self.passport_kwargs)
            elif self.arch == 'resnet':
                sub_model = ResNet18Private(num_classes=self.num_classes, passport_kwargs=self.passport_kwargs)
            else:
                raise NotImplementedError
            sub_model = sub_model.to(device)

            if self.arch == 'resnet':
                for m in self.model.modules():
                    if isinstance(m, PassportPrivateBlock):
                        keyname = 'key_private'
                        skeyname = 'skey_private'
                        bname = 'b'
                        fcname = 'fc'
                        
                        key, skey = m.__getattr__(keyname).data.clone(), m.__getattr__(skeyname).data.clone()

                        b = m.__getattr__(bname).data.clone()
                        m.__delattr__(keyname) 
                        m.__delattr__(skeyname)
                        m.__delattr__(fcname)
                        m.__delattr__('sign_loss_private')

                passport_settings = self.passport_config
                for l_key in passport_settings:  # layer
                    if isinstance(passport_settings[l_key], dict):
                        for i in passport_settings[l_key]:  # sequential
                            for m_key in passport_settings[l_key][i]:  # convblock
                                flag = passport_settings[l_key][i][m_key]
                                tl_m = sub_model.__getattr__(l_key)[int(i)].__getattr__(m_key)  # type: ConvBlock
                                self_m = self.model.__getattr__(l_key)[int(i)].__getattr__(m_key)
                                if not flag:
                                    tl_m.load_state_dict(self_m.state_dict(), (not flag))
                                    for param in tl_m.parameters():
                                        param.requires_grad = False
                                else:
                                    tl_m.conv.weight.data.copy_(self_m.conv.weight.data)
                                    tl_m.conv.weight.requires_grad = False

                                    tl_m.scale0.data.copy_(self_m.scale0.data)
                                    tl_m.scale1.data.copy_(self_m.scale1.data)
                                    tl_m.bias0.data.copy_(self_m.bias0.data)
                                    tl_m.bias1.data.copy_(self_m.bias1.data)
                                    tl_m.bn0 = self_m.bn0

                    else:
                        print("FFFFFFFFFFFFFFFFFFFFFFF")
                        tl_m = sub_model.__getattr__(l_key)
                        self_m = self.model.__getattr__(l_key)
                        flag = passport_settings[l_key]
                        tl_m.load_state_dict(self_m.state_dict(), (not flag))
                        for param in tl_m.parameters():
                            param.requires_grad = False

                l_key = 'linear'
                tl_m = sub_model.__getattr__(l_key)
                self_m = self.model.__getattr__(l_key)
                tl_m.load_state_dict(self_m.state_dict(), True)
                for param in tl_m.parameters():
                    param.requires_grad = False

            elif self.arch == 'alexnet':

                for m in self.model.modules():
                    if isinstance(m, PassportPrivateBlock):
                        keyname = 'key_private'
                        skeyname = 'skey_private'
                        bname = 'b'
                        b = m.__getattr__(bname).data.clone()
                        m.__delattr__(skeyname)
                            
                for tl_m, self_m in zip(sub_model.features, self.model.features):
                    if isinstance(self_m, PassportPrivateBlock):
                        tl_m.conv.weight.data.copy_(self_m.conv.weight.data)
                        tl_m.conv.weight.requires_grad = False

                        tl_m.scale0.data.copy_(self_m.scale0.data)
                        tl_m.scale1.data.copy_(self_m.scale1.data)
                        tl_m.bias0.data.copy_(self_m.bias0.data)
                        tl_m.bias1.data.copy_(self_m.bias1.data)
                        tl_m.bn0 = self_m.bn0
                    else:
                        tl_m.load_state_dict(self_m.state_dict(), True)
                        for param in tl_m.parameters():
                            param.requires_grad = False

                sub_model.classifier.load_state_dict(self.model.classifier.state_dict(), True)
                for param in sub_model.classifier.parameters():
                    param.requires_grad = False

            else:
                raise NotImplementedError

            key_x, x_inds = passport_generator.get_key_sig_att(att_data_x, 1)
            key_y, y_inds = passport_generator.get_key_sig_att(att_data_y, 1)
            key_x, key_y = key_x.to(self.device), key_y.to(self.device)
            passport_generator.set_key(sub_model, sub_model, key_x, key_y)

            if self.arch == 'resnet':
                passport_settings = self.passport_config
                for l_key in passport_settings:  # layer
                    if isinstance(passport_settings[l_key], dict):
                        for i in passport_settings[l_key]:  # sequential
                            for m_key in passport_settings[l_key][i]:  # convblock
                                flag = passport_settings[l_key][i][m_key]
                                tl_m = sub_model.__getattr__(l_key)[int(i)].__getattr__(m_key)  # type: ConvBlock
                                self_m = self.model.__getattr__(l_key)[int(i)].__getattr__(m_key)
                                if flag:
                                    b = tl_m.__getattr__('b').data.clone()
                                    b_ = self_m.__getattr__('b').data.clone()
                                    fakepassport.append(b.to(device))
                                    origpassport.append(b_.to(device))
                    else:
                        tl_m = sub_model.__getattr__(l_key)
                        self_m = self.model.__getattr__(l_key)
                        flag = passport_settings[l_key]
                        if flag:
                            b = tl_m.__getattr__('b').data.clone()
                            b_ = self_m.__getattr__('b').data.clone()
                            fakepassport.append(b.to(device))
                            origpassport.append(b_.to(device))

            elif self.arch == 'alexnet':
                for tl_m, self_m in zip(sub_model.features, self.model.features):
                    if isinstance(self_m, PassportPrivateBlock):
                        b = tl_m.__getattr__('b').data.clone()
                        b_ = self_m.__getattr__('b').data.clone()
                        fakepassport.append(b.to(device))
                        origpassport.append(b_.to(device))

            trainable_params = sum(p.numel() for p in sub_model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {trainable_params}")
            
            #optimizer = torch.optim.SGD(sub_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, sub_model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005)
            #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005)

            scheduler = None
            criterion = nn.CrossEntropyLoss()
            history = []
            
            print('-----------------------Before training------------------------')
            print('Before training', file = lf)
            res = {}
            cs = []
            bit_num = 0
            with torch.no_grad():
                for d1, d2 in zip(origpassport, fakepassport):
                    d1 = d1.view(d1.size(0), -1)
                    d2 = d2.view(d2.size(0), -1)
                    cs.append((d1 != d2).sum().item())
                    # print('d1', d1.size())
                    # print('d2', d2.size())
                    bit_num += len(d1)
                    #cs.append(F.cosine_similarity(d1, d2).item())
            print(f'Bit error rate of Real and Fake signature: {sum(cs) / bit_num:.4f}')
            f = open(best_file,'a')
            f.write('Bit error rate of Real and Fake signature:' + '\n')
            f.write(str(sum(cs) / bit_num) + '\n')
            f.flush()


            valres = test_fake(sub_model, criterion, self.valid_data, device)
            #test(self.model, device, self.valid_data, msg='Pri_pub Result', ind=1)


            for key in valres: res[f'valid_{key}'] = valres[key]

            print(res)
            print(res, file=lf)
            # sys.exit(0)
            history.append(res)

            best_acc = 0
            best_ep = 0
            best_sign_acc = 0
            best_bal_acc = 0

            for ep in range(1, epochs + 1):
                if scheduler is not None:
                    scheduler.step()

                print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
                print(f'Epoch {ep:3d}:')
                print(f'Epoch {ep:3d}:',file=lf)
                print('Training for ERB ambigious')
                #trainres = train_maximize(origpassport, fakepassport, sub_model, optimizer, criterion, self.train_data, device, self.type)
                trainres = train_ERB(sub_model, optimizer, criterion, self.train_data, device, self.type, ep)
                #trainres = train_ERB(self.model, optimizer, criterion, self.train_data, device, self.type, ep)

                print('Testing')
                print('Testing', file=lf)
                valres = test_fake(sub_model, criterion, self.valid_data, device)
                #valres = test_fake(self.model, criterion, self.valid_data, device)

                print(valres, file=lf)
                print('\n', file=lf)

                if best_acc < valres['foreacc']:
                    print(f'Found best at epoch {ep}\n')
                    best_acc = valres['foreacc']
                    best_ep = ep
                f = open(best_file,'a')
                f.write(str(best_acc) + '\n')
                f.write("best epoch: %s"%str(best_ep) + '\n')
                f.flush()

                if best_sign_acc < valres['signacc']:
                    print(f'Found best sign acc at epoch {ep}\n')
                    best_sign_acc = valres['signacc']
                    best_ep = ep
                f = open(best_file,'a')
                f.write(str(best_sign_acc) + '\n')
                f.write("best sign epoch: %s"%str(best_ep) + '\n')
                f.flush()

                if best_bal_acc < valres['baldis']:
                    print(f'Found best bal acc at epoch {ep}\n')
                    best_bal_acc = valres['baldis']
                    best_ep = ep
                f = open(best_file,'a')
                f.write(str(best_bal_acc) + '\n')
                f.write("best bal epoch: %s"%str(best_ep) + '\n')
                f.flush()

                res = {}

                for key in trainres: res[f'train_{key}'] = trainres[key]
                for key in valres: res[f'valid_{key}'] = valres[key]
                res['epoch'] = ep

                history.append(res)

                torch.save({'state_dict': sub_model.state_dict()},
                            f'{logdir}/{self.arch}-v-pro-fore-ERb-last-{self.dataset}-{self.type}-e{ep}.pth')

                histdf = pd.DataFrame(history)
                wandb.log({
                "sign_acc": valres["signacc"] * 100, 
                "Dep. acc": valres["depacc"] * 100, 
                "Fore. acc": valres["foreacc"] * 100, 
                #"bal_dis": valres["baldis"]* 100, 
                })
            histdf.to_csv(f'{logdir}/{self.arch}-v-pro-fore-ERb-last-history-{self.dataset}-{self.type}.csv')
            return


