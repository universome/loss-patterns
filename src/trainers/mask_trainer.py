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

from src.models import MaskModel, SimpleModel
from src.utils import validate, validate_weights, orthogonalize


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
        self.model = MaskModel(self.mask, self.config.hp.scaling).to(self.config.firelab.device_name)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_optimizers(self):
        self.optim = Adam(self.model.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        x = batch[0].to(self.config.firelab.device_name)
        y = batch[1].to(self.config.firelab.device_name)

        i, j = self.model.sample_idx()
        preds = self.model.run_from_weights(self.model.cell_center(i,j), x)
        clf_loss = self.criterion(preds, y).mean()
        acc = (preds.argmax(dim=1) == y).float().mean()

        if self.mask[i][j] == 0:
            loss = -clf_loss.clamp(0, self.config.hp.clip_threshold) * self.config.hp.negative_loss_coef
        else:
            loss = clf_loss

        ort_reg, norm_reg = self.model.compute_reg()
        right_len = self.model.lower_right.norm()

        final_loss = loss + self.config.hp.ort_reg_coef * ort_reg + self.config.hp.norm_reg_coef * norm_reg

        self.optim.zero_grad()
        final_loss.backward()
        self.optim.step()

        if self.mask[i][j] == 0:
            self.writer.add_scalar('bad/train/loss', clf_loss.item(), self.num_iters_done)
            self.writer.add_scalar('bad/train/acc', acc.item(), self.num_iters_done)
        elif self.mask[i][j] == 1:
            self.writer.add_scalar('good/train/loss', clf_loss.item(), self.num_iters_done)
            self.writer.add_scalar('good/train/acc', acc.item(), self.num_iters_done)
        else:
            raise NotImplementedError

        self.writer.add_scalar('Reg/ort', ort_reg.item(), self.num_iters_done)
        self.writer.add_scalar('Reg/norm', norm_reg.item(), self.num_iters_done)
        self.writer.add_scalar('Reg/right_len', right_len.item(), self.num_iters_done)

    def on_training_done(self):
        self.visualize_minimum()

    def compute_mask_scores(self):
        e1 = self.model.upper_left.to(self.config.firelab.device_name)
        e2 = orthogonalize(self.model.lower_right, e1, adjust_len=True)

        ts = np.linspace(0, max(self.mask.shape) * 2, num=30)
        ss = np.linspace(0, max(self.mask.shape) * 2, num=30)

        dummy_model = SimpleModel().to(self.config.firelab.device_name)
        weights = [[self.model.lower_left + t * e1 + s * e2 for s in ss] for t in ts]
        scores = [[validate_weights(w, self.val_dataloader, dummy_model) for w in w_row] for w_row in tqdm(weights)]

        return ss, ts, scores

    def visualize_minimum(self):
        ss, ts, scores = self.compute_mask_scores()
        fig = self.build_minimum_figure(ss, ts, scores)
        self.writer.add_figure('Minimum', fig, self.num_iters_done)

    def build_minimum_figure(self, ss, ts, scores):
        X, Y = np.meshgrid(ss, ts)

        fig = plt.figure(figsize=(20, 4))

        plt.subplot(141)
        cntr = plt.contourf(X, Y, [[s[0] for s in s_line] for s_line in scores],
                            cmap="RdBu_r", levels=np.linspace(0.3, 2.5, 30))
        plt.title('Loss [test]')
        plt.colorbar(cntr)

        plt.subplot(142)
        cntr = plt.contourf(X, Y, [[s[1] for s in s_line] for s_line in scores],
                            cmap="RdBu_r", levels=np.linspace(0.6, 0.9, 30))
        plt.title('Accuracy [test]')
        plt.colorbar(cntr)

        plt.subplot(143)
        cntr = plt.contourf(X, Y, [[s[0] for s in s_line] for s_line in scores],
                            cmap="RdBu_r", levels=100)
        plt.title('Loss [test]')
        plt.colorbar(cntr)

        plt.subplot(144)
        cntr = plt.contourf(X, Y, [[s[1] for s in s_line] for s_line in scores],
                            cmap="RdBu_r", levels=100)
        plt.title('Accuracy [test]')
        plt.colorbar(cntr)

        return fig

    def validate(self):
        self.model.is_good_mode = True
        good_val_loss, good_val_acc = validate(self.model, self.train_dataloader, self.criterion)
        self.model.is_good_mode = False
        bad_val_loss, bad_val_acc = validate(self.model, self.train_dataloader, self.criterion)
        self.model.is_good_mode = True

        self.writer.add_scalar('good/val/loss', good_val_loss, self.num_iters_done)
        self.writer.add_scalar('good/val/acc', good_val_acc, self.num_iters_done)
        self.writer.add_scalar('bad/val/loss', bad_val_loss, self.num_iters_done)
        self.writer.add_scalar('bad/val/acc', bad_val_acc, self.num_iters_done)
