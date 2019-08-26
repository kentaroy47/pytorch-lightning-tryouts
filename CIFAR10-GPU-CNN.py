#!/usr/bin/env python
# coding: utf-8

# # use CIFAR 10 to train CNNs with Pytorch-lightning!
# 
# let's develop 16FP and multi-GPU training.. :)

# In[1]:


# import stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models import *
# from utils import progress_bar

from test_tube import HyperOptArgumentParser, Experiment

torch.backends.cudnn.benchmark = False


# In[2]:


# imports for Lightning Modules
import os
from collections import OrderedDict
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule


# # Lightning Modelを定義する。
# おまじない的な側面も強い。Tutorialを読むのを推奨する。

# In[3]:


# define lightning model
class LightningModel(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(LightningModel, self).__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand(5, 3, 32, 32)

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        # inlayer
        self.c1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.c2 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # 16x16
        self.c3 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.c4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # 8x8
               
        self.c_d1 = nn.Linear(in_features=8*8*64,
                              out_features=self.hparams.hidden_dim)
        self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
        self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)
        
        self.c_d2 = nn.Linear(in_features=self.hparams.hidden_dim,
                              out_features=self.hparams.out_features)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool1(x) # 16
        
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.pool2(x) # 8
    
        batch_size = x.size(0)     
        x = F.relu(self.c_d1(x.view(batch_size, -1)))
        #x = self.c_d1_bn(x)
        #x = self.c_d1_drop(x)
        
        x = self.c_d2(x)
        logits = F.log_softmax(x, dim=1)

        return logits
    
    # criterion的にロスを定義する？
    def loss(self, labels, logits):
        nll = F.nll_loss(logits, labels)
        return nll
    
    # 学習のstepで何をやるか定義
    def training_step(self, data_batch, batch_i):
        """
        Lightning calls this inside the training loop
        :param data_batch:
        :return:
        """
        # forward pass
        x, y = data_batch
        #x = x.view(x.size(0), -1) #全結合のためflatten

        y_hat = self.forward(x) # forward

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({
            'loss': loss_val
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    # valで何をやるか定義する
    def validation_step(self, data_batch, batch_i):
        """
        Lightning calls this inside the validation loop
        :param data_batch:
        :return:
        """
        x, y = data_batch
        #x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': val_acc,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dic = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        return tqdm_dic

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    # ここでデータローダーを定義する！
    def __dataloader(self, train):
        # init data generators
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if train:
            dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        else:
            dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        # when using multi-node (ddp) we need to add the datasampler
        train_sampler = None
        batch_size = self.hparams.batch_size

        if self.trainer.use_ddp:
            train_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
            batch_size = batch_size // self.trainer.world_size  # scale batch size

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler
        )

        return loader

    @pl.data_loader
    def tng_dataloader(self):
        print('tng data loader called')
        return self.__dataloader(train=True)

    @pl.data_loader
    def val_dataloader(self):
        print('val data loader called')
        return self.__dataloader(train=False)

    @pl.data_loader
    def test_dataloader(self):
        print('test data loader called')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip=5.0)

        # network params
        parser.add_argument('--in_features', default=3 * 32 * 32, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--hidden_dim', default=5000, type=int)
        parser.opt_list('--drop_prob', default=0.2, options=[0.2, 0.5], type=float, tunable=False)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        # training params (opt)
        parser.opt_list('--learning_rate', default=0.001 * 8, type=float,
                        options=[0.0001, 0.0005, 0.001, 0.005],
                        tunable=False)
        parser.opt_list('--optimizer_name', default='adam', type=str,
                        options=['adam'], tunable=False)

        # if using 2 nodes with 4 gpus each the batch size here
        #  (256) will be 256 / (2*8) = 16 per gpu
        parser.opt_list('--batch_size', default=256, type=int,
                        options=[32, 64, 128, 256], tunable=False,
                        help='batch size will be divided over all gpus being used across all nodes')
        return parser


# # argparseをLightningModelに渡す

# In[4]:


# although we user hyperOptParser, we are using it only as argparse right now
parent_parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)

# dirs
root_dir = os.path.dirname(os.path.realpath("."))
demo_log_dir = os.path.join(root_dir, 'pt_lightning_demo_logs')
checkpoint_dir = os.path.join(demo_log_dir, 'model_weights')
test_tube_dir = os.path.join(demo_log_dir, 'test_tube_data')

# gpu args
parent_parser.add_argument('--gpus', type=str, default='-1',
                               help='how many gpus to use in the node.'
                                    'value -1 uses all the gpus on the node')
parent_parser.add_argument('--test_tube_save_path', type=str,
                           default=test_tube_dir, help='where to save logs')
parent_parser.add_argument('--model_save_path', type=str,
                           default=checkpoint_dir, help='where to save model')
parent_parser.add_argument('--experiment_name', type=str,
                           default='pt_lightning_exp_a', help='test tube exp name')

# allow model to overwrite or extend args
parser = LightningModel.add_model_specific_args(parent_parser, root_dir)
hyperparams = parser.parse_args(args=[]) # for jupyter


# # モデルを初期化

# In[5]:


# init lightning model
print("load model")
model = LightningModel(hyperparams)
print("build model")


# # Experimentを初期化

# In[6]:


# Init experiment
exp = Experiment(
        name=hyperparams.experiment_name,
        save_dir=hyperparams.test_tube_save_path,
        autosave=False,
        description='test demo'
    )


# # Define Callbacks
# Kerasのように定義できる！

# In[7]:


model_save_path = '{}/{}/{}'.format(hyperparams.model_save_path, exp.name, exp.version)
early_stop = EarlyStopping(
        monitor='val_acc',
        patience=3,
        verbose=True,
        mode='max'
    )

checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )


# # Trainerを初期化する

# In[8]:


trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=hyperparams.gpus
    )


# # 学習を開始！

# In[9]:


trainer.fit(model)


# In[ ]:


import torch.nn as nn
class CNN(nn.Module): 
    def __init__(self, bs):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(CNN, self).__init__()
        

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = torch.rand(5, 3, 32, 32)

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        # inlayer
        self.c1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.c2 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # 16x16
        self.c3 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.c4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) # 8x8
               
        self.c_d1 = nn.Linear(in_features=8*8*64,
                              out_features=128)
        self.c_d1_bn = nn.BatchNorm1d(128)
        self.c_d1_drop = nn.Dropout(0.5)
        
        self.c_d2 = nn.Linear(in_features=128,
                              out_features=10)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool1(x) # 16
        
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = self.pool2(x) # 8
    
        batch_size = x.size(0)     
        x = F.relu(self.c_d1(x.view(batch_size, -1)))
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        
        x = self.c_d2(x)
        logits = F.log_softmax(x, dim=1)

        return logits
nn = CNN(8)
print(nn)


# In[ ]:




