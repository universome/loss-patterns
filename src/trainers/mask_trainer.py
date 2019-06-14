import os
import math
import time
import random
from itertools import chain
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
from firelab import BaseTrainer
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread

from src.models import MaskModel, SimpleModel, SimpleModelOperation
from src.models.vgg import VGG11, VGG11Operation
from src.utils import validate, validate_weights, orthogonalize, weight_to_param, param_sizes


class MaskTrainer(BaseTrainer):
    def __init__(self, config):
        super(MaskTrainer, self).__init__(config)

        if self.config.mask_type == 'icon':
            project_path = self.config.firelab.project_path
            data_dir = os.path.join(project_path, self.config.data_dir)
            icon = imread(os.path.join(data_dir, self.config.hp.icon_file_path))
            self.mask = np.array(icon > 0).astype(np.float)
        elif self.config.mask_type == 'custom':
            self.mask = np.array(self.config.mask)
        elif self.config.mask_type == 'square':
            self.mask = generate_square_mask(self.config.hp.square_size)
            self.mask = make_mask_ternary(self.mask)
        else:
            raise NotImplementedError('Mask type %s is not supported' % self.config.mask_type)

        if self.config.model_name == "vgg":
            self.torch_model_cls = VGG11
            self.model_op_cls = VGG11Operation
        elif self.config.model_name == "simple":
            self.torch_model_cls = SimpleModel
            self.model_op_cls = SimpleModelOperation
        else:
            raise NotImplementedError("Model %s is not supported" % self.config.model_name)

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        data_dir = os.path.join(project_path, self.config.data_dir)

        data_train = FashionMNIST(data_dir, download=True, train=True, transform=ToTensor())
        data_test = FashionMNIST(data_dir, download=True, train=False, transform=ToTensor())

        self.train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=3, shuffle=True)
        self.val_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=3, shuffle=False)

    def before_training_start_hook(self):
        self.plot_mask()
        self.plot_all_weights_histograms()

    def init_models(self):
        self.model = MaskModel(self.mask, self.torch_model_cls, self.model_op_cls, self.config.hp.scaling)
        self.model = self.model.to(self.config.firelab.device_name)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_optimizers(self):
        self.optim = Adam(self.model.parameters(), lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        x = batch[0].to(self.config.firelab.device_name)
        y = batch[1].to(self.config.firelab.device_name)

        good_losses = []
        good_accs = []
        bad_losses = []
        bad_accs = []

        good_idx = self.model.get_class_idx(1).tolist()
        bad_idx = self.model.get_class_idx(-1).tolist()

        num_good_points_to_use = min(len(good_idx), self.config.hp.num_cells_in_update.good)
        num_bad_points_to_use = min(len(bad_idx), self.config.hp.num_cells_in_update.bad)

        for i, j in random.sample(good_idx, num_good_points_to_use):
            preds = self.model.run_from_weights(self.model.cell_center(i,j), x)
            good_losses.append(self.criterion(preds, y).mean())
            good_accs.append((preds.argmax(dim=1) == y).float().mean())

        for i, j in random.sample(bad_idx, num_bad_points_to_use):
            preds = self.model.run_from_weights(self.model.cell_center(i,j), x)
            bad_losses.append(self.criterion(preds, y).mean())
            bad_accs.append((preds.argmax(dim=1) == y).float().mean())

        good_losses = torch.stack(good_losses)
        good_accs = torch.stack(good_accs)
        bad_losses = torch.stack(bad_losses)
        bad_accs = torch.stack(bad_accs)

        good_loss = good_losses.mean()
        bad_loss = bad_losses.clamp(0, self.config.hp.clip_threshold).mean()
        final_loss = good_loss - self.config.hp.negative_loss_coef * bad_loss
        # final_loss += self.config.hp.ort_reg_coef * ort_reg + self.config.hp.norm_reg_coef * norm_reg

        self.optim.zero_grad()
        final_loss.backward()
        clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_threshold)
        self.optim.step()

        self.writer.add_scalar('bad/train/loss', bad_losses.mean().mean().item(), self.num_iters_done)
        self.writer.add_scalar('bad/train/acc', bad_accs.mean().item(), self.num_iters_done)
        self.writer.add_scalar('good/train/loss', good_losses.mean().item(), self.num_iters_done)
        self.writer.add_scalar('good/train/acc', good_accs.mean().item(), self.num_iters_done)

        # Tracking stats
        self.writer.add_scalar('Stats/ort', torch.dot(self.model.right, self.model.up).abs().item(), self.num_iters_done)
        # self.writer.add_scalars('lengths', {
        #     'right': self.model.right.norm(),
        #     'up': self.model.up.norm(),
        # }, self.num_iters_done)
        self.writer.add_scalar('Stats/lengths/right', self.model.right.norm(), self.num_iters_done)
        self.writer.add_scalar('Stats/lengths/up', self.model.up.norm(), self.num_iters_done)

        self.writer.add_scalar('Stats/grad_norms/origin', self.model.origin.grad.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/grad_norms/right_param', self.model.right_param.grad.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/grad_norms/up_param', self.model.up_param.grad.norm().item(), self.num_iters_done)

    def on_training_done(self):
        self.visualize_minimum()

    def compute_mask_scores(self):
        start = time.time()

        e1 = self.model.up.to(self.config.firelab.device_name)
        e2 = self.model.right.to(self.config.firelab.device_name)

        ts = self.config.hp.scaling * np.linspace(-1, max(self.mask.shape), num=20)
        ss = self.config.hp.scaling * np.linspace(-1, max(self.mask.shape), num=20)

        dummy_model = self.torch_model_cls().to(self.config.firelab.device_name)
        scores = [[validate_weights(self.model.origin + t * e1 + s * e2, self.val_dataloader, dummy_model) for s in ss] for t in tqdm(ts)]

        print('Scoring took', time.time() - start)

        return ss, ts, scores

    def visualize_minimum(self):
        ss, ts, scores = self.compute_mask_scores()
        fig = self.build_minimum_figure(ss, ts, scores)
        self.writer.add_figure('Minimum', fig, self.num_iters_done)

    def build_minimum_figure(self, ss, ts, scores):
        X, Y = np.meshgrid(ss, ts)

        fig = plt.figure(figsize=(20, 4))

        plt.subplot(141)
        cntr = plt.contourf(X, Y, [[s[0] for s in s_line] for s_line in scores], cmap="RdBu_r", levels=np.linspace(0.3, 2.5, 30))
        plt.title('Loss [test]')
        plt.colorbar(cntr)

        plt.subplot(142)
        cntr = plt.contourf(X, Y, [[s[1] for s in s_line] for s_line in scores], cmap="RdBu_r", levels=np.linspace(0.6, 0.9, 30))
        plt.title('Accuracy [test]')
        plt.colorbar(cntr)

        plt.subplot(143)
        cntr = plt.contourf(X, Y, [[s[0] for s in s_line] for s_line in scores], cmap="RdBu_r", levels=100)
        plt.title('Loss [test]')
        plt.colorbar(cntr)

        plt.subplot(144)
        cntr = plt.contourf(X, Y, [[s[1] for s in s_line] for s_line in scores], cmap="RdBu_r", levels=np.linspace(0, 1, 100))
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

        self.plot_all_weights_histograms()

    def plot_mask(self):
        fig = plt.figure(figsize=(5, 5))
        mask_img = np.copy(self.mask)
        mask_img[mask_img == 2] = 0.5
        plt.imshow(mask_img, cmap='gray')
        self.writer.add_figure('Mask', fig, self.num_iters_done)

    def plot_params_histograms(self, w, subtag:str):
        dummy_model = self.torch_model_cls()
        params = weight_to_param(w, param_sizes(dummy_model.parameters()))
        tags = ['Weights_hist/{}/{}'.format(i, subtag) for i in range(len(params))]

        for tag, param in zip(tags, params):
            self.writer.add_histogram(tag, param, self.num_iters_done)

    def plot_all_weights_histograms(self):
        self.plot_params_histograms(self.model.origin, 'origin')
        self.plot_params_histograms(self.model.right, 'right')
        self.plot_params_histograms(self.model.up, 'up')
        self.plot_params_histograms(self.model.origin + self.model.right, 'origin+right')
        self.plot_params_histograms(self.model.origin + self.model.up, 'origin+up')


def generate_square_mask(square_size):
    assert square_size >= 3

    mask = np.zeros((square_size, square_size))
    mask[1:-1, 1] = 1
    mask[1:-1, -2] = 1
    mask[1, 1:-1] = 1
    mask[-2, 1:-1] = 1

    return mask


def make_mask_ternary(mask):
    """
    Takes 0/1 mask and makes -1/0/1 mask, setting 0 to -1,
    and those -1, which are far away from 1 to 0 (so we do not look at them)
    """
    useless_zeros:List[Tuple[int, int]] = []

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 1: continue

            num_ones = mask[max(i-1, 0):i+2, max(j-1, 0):j+2].sum()

            if num_ones == 0:
                useless_zeros.append((i, j))

    result = np.copy(mask)
    result[[i for i,j in useless_zeros], [j for i,j in useless_zeros]] = 2

    # Convert 0/1/2 mask to -1/0/1 mask, because it's more sensible
    result[result == 0] = -1
    result[result == 2] = 0

    return result
