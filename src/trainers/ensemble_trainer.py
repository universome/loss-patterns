import os
import shutil
import time
import random
from itertools import chain
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.clip_grad import clip_grad_norm_
from firelab import BaseTrainer
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread

from src.models import EnsembleModel, ConvModel
from src.models.vgg import VGG11
from src.utils import validate, validate_weights, weight_to_param, param_sizes
from .mask_trainer import MaskTrainer

class EnsembleTrainer(BaseTrainer):
    def __init__(self, config):
        super(EnsembleTrainer, self).__init__(config)

    def init_models(self):
        MaskTrainer.init_torch_model_builder(self)

        self.model = EnsembleModel(
            self.torch_model_builder,
            self.config.hp.num_models,
            self.config.hp.coords_init_strategy
        )
        self.model = self.model.to(self.device_name)

    def init_dataloaders(self):
        MaskTrainer.init_dataloaders(self)

    def init_criterions(self):
        MaskTrainer.init_criterions(self)

    def init_optimizers(self):
        MaskTrainer.init_optimizers(self)

    def train_on_batch(self, batch):
        self.model.train()
        self.optim.zero_grad()

        x = batch[0].to(self.device_name)
        y = batch[1].to(self.device_name)

        point_losses = []
        point_accs = []
        point_preds = []
        point_coords = list(range(self.model.coords.size(0)))
        num_models = min(len(point_coords), self.config.hp.num_models_per_update)

        for i in random.sample(point_coords, num_models):
            preds = self.model.run_model_by_id(i, x)

            loss = self.criterion(preds, y).mean()
            loss /= num_models
            loss.backward() # To make current graph free

            point_losses.append(loss.item())
            point_accs.append((preds.argmax(dim=1) == y).float().mean().item())
            point_preds.append(preds)

        point_accs = np.array(point_accs)

        self.writer.add_scalar('Train/loss/mean', np.mean(point_losses), self.num_iters_done)
        self.writer.add_scalar('Train/loss/max', np.max(point_losses), self.num_iters_done)
        self.writer.add_scalar('Train/loss/min', np.min(point_losses), self.num_iters_done)

        self.writer.add_scalar('Train/acc/mean', np.mean(point_accs), self.num_iters_done)
        self.writer.add_scalar('Train/acc/max', np.max(point_accs), self.num_iters_done)
        self.writer.add_scalar('Train/acc/min', np.min(point_accs), self.num_iters_done)

        # Adding ensemble decorrelation regularization
        # TODO:
        # - KL, MSE, other divergences on model outputs
        # - adversarial loss to guess from which model prediction has came
        # - just make weights farther from each other
        # - correlation/MSE for losses between models (yes, it will be differentiable)

        self.writer.add_scalar('Stats/grad_norms/coords', self.model.coords.grad.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/grad_norms/origin_param', self.model.origin_param.grad.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/grad_norms/right_param', self.model.right_param.grad.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/grad_norms/up_param', self.model.up_param.grad.norm().item(), self.num_iters_done)

        clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_threshold)
        self.optim.step()

        self.writer.add_histogram('Coords/x', self.model.coords[:,0].cpu().detach().numpy(), self.num_iters_done)
        self.writer.add_histogram('Coords/y', self.model.coords[:,1].cpu().detach().numpy(), self.num_iters_done)
        self.writer.add_scalar('Stats/norms/origin_param', self.model.origin_param.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/norms/right_param', self.model.right_param.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/norms/up_param', self.model.up_param.norm().item(), self.num_iters_done)

    def validate(self):
        guessed = np.array([])
        losses = np.array([])

        self.model.eval()

        with torch.no_grad():
            for x, y in self.val_dataloader:
                x, y = x.to(self.device_name), y.to(self.device_name)
                preds = self.model(x)
                loss = self.criterion(preds, y)
                losses = np.hstack([losses, loss.cpu().numpy()])
                guessed = np.hstack([guessed, (preds.argmax(dim=1) == y).long().cpu().numpy()])

        self.writer.add_scalar('Val/loss', losses.mean(), self.num_epochs_done)
        self.writer.add_scalar('Val/acc', guessed.mean(), self.num_epochs_done)
