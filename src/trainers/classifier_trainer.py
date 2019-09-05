import os
import math
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from firelab import BaseTrainer
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import ConvModel
from src.models.resnet import FastResNet
from src.utils import validate, weight_vector
from src.trainers.mask_trainer import MaskTrainer
from src.models.layer_ops import convert_sequential_model_to_op


class ClassifierTrainer(BaseTrainer):
    def __init__(self, config):
        super(ClassifierTrainer, self).__init__(config)

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.paths.project_path
        data_dir = os.path.join(project_path, self.config.data_dir)

        train_transform = transforms.Compose([
            transforms.Pad(padding=4),
            transforms.RandomCrop(size=(32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1., 1.)), # Cut out 8x8 square
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            # transforms.Pad(padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data_train = CIFAR10(data_dir, train=True, transform=train_transform)
        data_test = CIFAR10(data_dir, train=False, transform=test_transform)

        self.train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=0, shuffle=True)
        self.val_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=0, shuffle=False)

    def init_models(self):
        if self.config.hp.model_name == 'conv':
            self.model = ConvModel(self.config.hp.conv_model_config)
        elif self.config.hp.model_name == 'fast_resnet':
            model = FastResNet(n_classes=10, n_input_channels=3).nn
            self.model = convert_sequential_model_to_op(weight_vector(model.parameters()), model, detach=True)

            assert len(weight_vector(model.parameters())) == len(weight_vector(self.model.parameters()))
        else:
            raise NotImplementedError(f'Model {self.config.hp.model_name} is not supported')

        self.model = self.model.to(self.device_name)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_optimizers(self):
        MaskTrainer.init_optimizers(self)

    def train_on_batch(self, batch):
        x = batch[0].to(self.device_name)
        y = batch[1].to(self.device_name)

        preds = self.model(x)
        loss = self.criterion(preds, y).sum()
        acc = (preds.argmax(dim=1) == y).float().mean()
        norm = weight_vector(self.model.parameters()).norm()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Train/loss', loss.item(), self.num_iters_done)
        self.writer.add_scalar('Train/acc', acc.item(), self.num_iters_done)
        self.writer.add_scalar('Stats/weights_norm', norm.item(), self.num_iters_done)

        if not self.scheduler is None:
            self.scheduler.step()
            self.writer.add_scalar('Stats/lr', self.scheduler.get_lr()[0], self.num_iters_done)

    def validate(self):
        self.model.eval()
        loss, acc = validate(self.model, self.val_dataloader, self.criterion)

        self.writer.add_scalar('Val/loss', loss.item(), self.num_epochs_done)
        self.writer.add_scalar('Val/acc', acc.item(), self.num_epochs_done)
