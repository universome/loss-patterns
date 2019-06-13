import os
import math
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from firelab import BaseTrainer
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor, ToPILImage, Compose
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import SimpleModel, VGG11
from src.utils import validate, validate_weights, weight_vector


class ClassifierTrainer(BaseTrainer):
    def __init__(self, config):
        super(ClassifierTrainer, self).__init__(config)

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        data_dir = os.path.join(project_path, self.config.data_dir)

        data_train = FashionMNIST(data_dir, download=True, train=True, transform=ToTensor())
        data_test = FashionMNIST(data_dir, download=True, train=False, transform=ToTensor())

        self.train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=3, shuffle=True)
        self.val_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=3, shuffle=False)

    def init_models(self):
        # self.model = SimpleModel().to(self.config.firelab.device_name)
        self.model = VGG11(n_input_channels=1, num_classes=10, head_size=512)
        self.model = self.model.to(self.config.firelab.device_name)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_optimizers(self):
        self.optim = Adam(self.model.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        x = batch[0].to(self.config.firelab.device_name)
        y = batch[1].to(self.config.firelab.device_name)

        preds = self.model(x)
        loss = self.criterion(preds, y).mean()
        acc = (preds.argmax(dim=1) == y).float().mean()
        norm = weight_vector(self.model.parameters()).norm()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Train/loss', loss.item(), self.num_iters_done)
        self.writer.add_scalar('Train/acc', acc.item(), self.num_iters_done)
        self.writer.add_scalar('Stats/weights_norm', norm.item(), self.num_iters_done)

    def validate(self):
        self.model.eval()
        loss, acc = validate(self.model, self.val_dataloader, self.criterion)

        self.writer.add_scalar('Val/loss', loss.item(), self.num_iters_done)
        self.writer.add_scalar('Val/acc', acc.item(), self.num_iters_done)
