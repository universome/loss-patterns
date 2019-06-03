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

from src.models import MaskModel
from src.utils import validate


class MaskTrainer(BaseTrainer):
    def __init__(self, config):
        super(MaskTrainer, self).__init__(config)

        self.mask = np.array(self.config.mask)


    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        data_dir = os.path.join(project_path, self.config.data_dir)

        data_train = FashionMNIST(data_dir, download=True, train=True, transform=ToTensor())
        data_test = FashionMNIST(data_dir, download=True, train=False, transform=ToTensor())

        self.train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=3, shuffle=True)
        self.val_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=3, shuffle=False)

    def init_models(self):
        self.model = MaskModel(self.mask).to(self.config.device_name)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_optimizers(self):
        self.optim = Adam(self.model.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        x = batch[0].to(self.config.device_name)
        y = batch[1].to(self.config.device_name)

        i, j = self.model.sample_idx()
        preds = self.model.run_from_weights(self.model.cell_center(i,j), x)
        loss = self.criterion(preds, y).mean()
        acc = (preds.argmax(dim=1) == y).float().mean()

        if self.mask[i][j] == 0:
            loss = -loss.clamp(0, self.config.hp.clip_threshold) * self.config.hp.negative_loss_coef

        ort_reg, norm_reg = self.model.compute_reg()
        right_len = self.model.lower_right.norm()

        final_loss = loss + self.config.hp.ort_reg_coef * ort_reg + self.config.hp.norm_reg_coef * norm_reg

        self.optim.zero_grad()
        final_loss.backward()
        self.optim.step()

        if self.mask[i][j] == 0:
            self.writer.add_scalar('Train/bad/loss', loss.item(), self.num_iters_done)
            self.writer.add_scalar('Train/bad/acc', acc.item(), self.num_iters_done)
        else:
            self.writer.add_scalar('Train/good/loss', loss.item(), self.num_iters_done)
            self.writer.add_scalar('Train/good/acc', acc.item(), self.num_iters_done)

        self.writer.add_scalar('Reg/ort', ort_reg.item(), self.num_iters_done)
        self.writer.add_scalar('Reg/norm', norm_reg.item(), self.num_iters_done)
        self.writer.add_scalar('Reg/right_len', right_len.item(), self.num_iters_done)

    def validate(self):
        self.model.is_good_mode = True
        good_val_loss, good_val_acc = validate(self.model, self.val_dataloader, self.criterion)
        self.model.is_good_mode = False
        bad_val_loss, bad_val_acc = validate(self.model, self.val_dataloader, self.criterion)
        self.model.is_good_mode = True

        self.writer.add_scalar('Val/good/loss', good_val_loss, self.num_iters_done)
        self.writer.add_scalar('Val/good/acc', good_val_acc, self.num_iters_done)
        self.writer.add_scalar('Val/bad/loss', bad_val_loss, self.num_iters_done)
        self.writer.add_scalar('Val/bad/acc', bad_val_acc, self.num_iters_done)
