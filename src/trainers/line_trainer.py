import os
import math
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from firelab import BaseTrainer
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models import LineModel
from src.utils import validate, validate_weights


class LineTrainer(BaseTrainer):
    def __init__(self, config):
        super(LineTrainer, self).__init__(config)

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        data_dir = os.path.join(project_path, self.config.data_dir)

        data_train = FashionMNIST(data_dir, download=True, train=True, transform=ToTensor())
        data_test = FashionMNIST(data_dir, download=True, train=False, transform=ToTensor())

        self.train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=3, shuffle=True)
        self.val_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=3, shuffle=False)

    def init_models(self):
        self.model = LineModel().to(self.config.device_name)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_optimizers(self):
        self.optim = Adam(self.model.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        x = batch[0].to(self.config.device_name)
        y = batch[1].to(self.config.device_name)

        preds = self.model(x)
        clf_loss = self.criterion(preds, y).mean()
        acc = (preds.argmax(dim=1) == y).float().mean()

        dist = (self.model.w_1 - self.model.w_2).norm().pow(2)
        w_1_len = self.model.w_1.norm().pow(2)
        w_2_len = self.model.w_2.norm().pow(2)

        if self.config.hp.dist_reg_coef == 0:
            final_loss = clf_loss
        else:
            final_loss = clf_loss + self.config.hp.dist_reg_coef * dist

        self.optim.zero_grad()
        final_loss.backward()
        self.optim.step()

        self.writer.add_scalar('Train/loss', clf_loss.item(), self.num_iters_done)
        self.writer.add_scalar('Train/acc', acc.item(), self.num_iters_done)
        self.writer.add_scalar('Reg/distance2', dist.item(), self.num_iters_done)
        self.writer.add_scalar('Reg/w_1_norm2', w_1_len.item(), self.num_iters_done)
        self.writer.add_scalar('Reg/w_2_norm2', w_2_len.item(), self.num_iters_done)

    def validate(self):
        self.model.eval()
        clf_loss, acc = validate(self.model, self.val_dataloader, self.criterion)

        self.writer.add_scalar('Val/loss', clf_loss.item(), self.num_iters_done)
        self.writer.add_scalar('Val/acc', acc.item(), self.num_iters_done)
