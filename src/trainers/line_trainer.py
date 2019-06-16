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
from tqdm import tqdm

from src.models import LineModel, SimpleModel
from src.utils import validate, validate_weights, linerp, get_weights_linerp
from src.utils import compute_weights_entropy_linerp, compute_activations_entropy_linerp
from src.plotting_utils import generate_linerp_plot, generate_acts_entropy_linerp_plot, generate_weights_entropy_linerp_plot


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
        self.model = LineModel().to(self.config.firelab.device_name)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_optimizers(self):
        self.optim = Adam(self.model.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        x = batch[0].to(self.config.firelab.device_name)
        y = batch[1].to(self.config.firelab.device_name)

        preds = self.model(x)
        clf_loss = self.criterion(preds, y).mean()
        acc = (preds.argmax(dim=1) == y).float().mean()

        dist = (self.model.w_1 - self.model.w_2).norm()
        w_1_len = self.model.w_1.norm()
        w_2_len = self.model.w_2.norm()

        if self.config.hp.dist_reg_coef == 0:
            final_loss = clf_loss
        else:
            final_loss = clf_loss - self.config.hp.dist_reg_coef * dist

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

    def after_training_hook(self):
        self.visualize_linerp()
        self.visualize_entropy_linerp()

    def visualize_linerp(self):
        dummy_model = SimpleModel().to(self.config.firelab.device_name)

        w1to2_linerp_scores_train = linerp(self.model.w_1, self.model.w_2, dummy_model, self.train_dataloader)
        w1to2_linerp_scores_test = linerp(self.model.w_1, self.model.w_2, dummy_model, self.val_dataloader)

        self.writer.add_figure('w_1 to w_2 linerp/accuracy', generate_linerp_plot(
            [s[1] for s in w1to2_linerp_scores_train],
            [s[1] for s in w1to2_linerp_scores_test],
        ))

        self.writer.add_figure('w_1 to w_2 linerp/loss', generate_linerp_plot(
            [s[0] for s in w1to2_linerp_scores_train],
            [s[0] for s in w1to2_linerp_scores_test],
        ))

    def visualize_entropy_linerp(self):
        # We are visualizing only between w_1 and w_2 here
        dummy_model = SimpleModel().to(self.config.firelab.device_name)

        # acts_ent_linerp_train = compute_activations_entropy_linerp(self.model.w_1, self.model.w_2, dummy_model.nn, self.train_dataloader)
        acts_ent_linerp_test = compute_activations_entropy_linerp(self.model.w_1, self.model.w_2, dummy_model.nn, self.val_dataloader)
        weights_ent_linerp = compute_weights_entropy_linerp(self.model.w_1, self.model.w_2)

        # self.writer.add_figure('activations_entropy_linerp/train', generate_acts_entropy_linerp_plot(acts_ent_linerp_train))
        self.writer.add_figure('activations_entropy_linerp/test', generate_acts_entropy_linerp_plot(acts_ent_linerp_test))
        self.writer.add_figure('weights_entropy_linerp', generate_weights_entropy_linerp_plot(weights_ent_linerp))

        for i, w in enumerate(get_weights_linerp(self.model.w_1, self.model.w_2)):
            self.writer.add_histogram('Weights histogram', w.detach().cpu(), i)
