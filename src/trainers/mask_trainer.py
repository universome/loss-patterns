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
import yaml

from src.models import MaskModel, SimpleModel, ConvModel
from src.models.vgg import VGG11
from src.utils import validate, validate_weights, weight_to_param, param_sizes


class MaskTrainer(BaseTrainer):
    def __init__(self, config):
        super(MaskTrainer, self).__init__(config)

        self.init_mask()

    def init_mask(self):
        if self.config.mask_type == 'icon':
            project_path = self.config.firelab.project_path
            data_dir = os.path.join(project_path, self.config.data_dir)
            icon = imread(os.path.join(data_dir, self.config.hp.icon_file_path))
            if self.config.hp.get('should_resize_icon', False):
                icon = resize(icon, self.config.hp.target_icon_size, mode='constant', anti_aliasing=True)
            icon = convert_img_to_binary(icon)
            self.mask = make_mask_ternary(icon)
        elif self.config.mask_type == 'custom':
            self.mask = np.array(self.config.mask)
        elif self.config.mask_type == 'square':
            self.mask = generate_square_mask(self.config.hp.square_size)
            self.mask = make_mask_ternary(self.mask)
        elif self.config.mask_type == 'randomly_filled_square':
            self.mask = generate_square_mask(self.config.hp.square_size)
            self.mask = randomly_fill_square(self.mask, self.config.hp.fill_prob)
            self.mask = make_mask_ternary(self.mask)
        elif self.config.mask_type == 'square_grid':
            self.mask = generate_square_grid_mask(self.config.hp.n_good_cells)
            self.mask = make_mask_ternary(self.mask)
        else:
            raise NotImplementedError('Mask type %s is not supported' % self.config.mask_type)

    def init_dataloaders(self):
        dataset = self.config.hp.get('dataset', 'FashionMNIST')
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        data_dir = os.path.join(project_path, self.config.data_dir)

        if dataset == 'FashionMNIST':
            data_train = FashionMNIST(data_dir, train=True, transform=transforms.ToTensor())
            data_test = FashionMNIST(data_dir, train=False, transform=transforms.ToTensor())
        elif dataset == 'MNIST':
            data_train = MNIST(data_dir, train=True, transform=transforms.ToTensor())
            data_test = MNIST(data_dir, train=False, transform=transforms.ToTensor())
        elif dataset == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            data_train = CIFAR10(data_dir, train=True, transform=transform)
            data_test = CIFAR10(data_dir, train=False, transform=transform)
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset}")

        data_vis_train = Subset(data_train, random.sample(range(len(data_train)), self.config.get('n_points_for_vis', 1000)))
        data_vis_test = Subset(data_test, random.sample(range(len(data_test)), self.config.get('n_points_for_vis', 1000)))

        self.train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
        self.vis_train_dataloader = DataLoader(data_vis_train, batch_size=batch_size, shuffle=False)
        self.vis_test_dataloader = DataLoader(data_vis_test, batch_size=batch_size, shuffle=False)

    def init_models(self):
        if self.config.model_name == "vgg":
            self.torch_model_builder = lambda: VGG11(
                n_input_channels=self.config.hp.get('n_input_channels', 1),
                use_bn=self.config.hp.get('use_bn', True))
        elif self.config.model_name == "simple":
            self.torch_model_builder = SimpleModel
        elif self.config.model_name == "conv":
            self.torch_model_builder = lambda: ConvModel(self.config.hp.conv_model_config).nn
        else:
            raise NotImplementedError("Model %s is not supported" % self.config.model_name)

        self.model = MaskModel(
            self.mask, self.torch_model_builder,
            should_center_origin=self.config.hp.should_center_origin,
            parametrization_type=self.config.hp.parametrization_type)
        self.model = self.model.to(self.config.firelab.device_name)

        # self.logger.info(f'Model initial orthogonality: {self.model.compute_ort_reg()}')
        # self.logger.info(f'Model params: {self.config.hp.conv_model_config.to_dict()}. Parametrization: {self.config.hp.parametrization_type}')

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_optimizers(self):
        optim_name = self.config.hp.get('optim', 'adam').lower()

        if optim_name == 'adam':
            self.optim = Adam(self.model.parameters(), lr=self.config.hp.lr)
        elif optim_name == 'sgd':
            self.optim = SGD(self.model.parameters(), lr=self.config.hp.lr, momentum=0.9)
        else:
            raise NotImplementedError(f'Unknown optimizer name: {optim_name}')

    def train_on_batch(self, batch):
        self.optim.zero_grad()

        x = batch[0].to(self.config.firelab.device_name)
        y = batch[1].to(self.config.firelab.device_name)

        good_losses = []
        good_accs = []
        bad_losses = []
        bad_accs = []

        good_idx = self.model.get_class_idx(1).tolist()
        bad_idx = self.model.get_class_idx(-1).tolist()

        num_good_points_to_use = min(len(good_idx), self.config.hp.num_good_cells_per_update)
        num_bad_points_to_use = min(len(bad_idx), self.config.hp.num_bad_cells_per_update)

        for i, j in random.sample(good_idx, num_good_points_to_use):
            preds = self.model.run_from_weights(self.model.compute_point(i,j), x)

            good_loss = self.criterion(preds, y).mean()
            good_losses.append(good_loss.item())
            good_loss /= num_good_points_to_use
            good_loss.backward() # To make the graph free

            good_accs.append((preds.argmax(dim=1) == y).float().mean().item())

        for i, j in random.sample(bad_idx, num_bad_points_to_use):
            preds = self.model.run_from_weights(self.model.compute_point(i,j), x)

            bad_loss = self.criterion(preds, y).mean()
            bad_losses.append(bad_loss.item())
            bad_loss = bad_loss.clamp(0, self.config.hp.neg_loss_clip_threshold)
            bad_loss /= num_bad_points_to_use
            bad_loss *= self.config.hp.negative_loss_coef
            bad_loss *= -1 # To make it grow
            bad_loss.backward() # To make the graph free

            bad_accs.append((preds.argmax(dim=1) == y).float().mean().item())

        good_losses = np.array(good_losses)
        good_accs = np.array(good_accs)
        bad_losses = np.array(bad_losses)
        bad_accs = np.array(bad_accs)

        # Adding regularization
        if self.config.hp.parametrization_type != "up_orthogonal":
            ort = self.model.compute_ort_reg()
            norm_diff = self.model.compute_norm_reg()
            reg_loss = self.config.hp.ort_l2_coef * ort.pow(2) + self.config.hp.norm_l2_coef * norm_diff.pow(2)
            reg_loss.backward()

            self.writer.add_scalar('Reg/ort', ort.item(), self.num_iters_done)
            self.writer.add_scalar('Reg/norm_diff', norm_diff.item(), self.num_iters_done)

        clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_threshold)
        self.optim.step()

        self.writer.add_scalar('good/train/loss', good_losses.mean().item(), self.num_iters_done)
        self.writer.add_scalar('good/train/acc', good_accs.mean().item(), self.num_iters_done)
        self.writer.add_scalar('bad/train/loss', bad_losses.mean().item(), self.num_iters_done)
        self.writer.add_scalar('bad/train/acc', bad_accs.mean().item(), self.num_iters_done)
        self.writer.add_scalar('diff/train/loss', good_losses.mean().item() - bad_losses.mean().item(), self.num_iters_done)
        self.writer.add_scalar('diff/train/acc', good_accs.mean().item() - bad_accs.mean().item(), self.num_iters_done)

        self.writer.add_scalar('Stats/lengths/right', self.model.right.norm(), self.num_iters_done)
        self.writer.add_scalar('Stats/lengths/up', self.model.up.norm(), self.num_iters_done)
        self.writer.add_scalar('Stats/grad_norms/origin', self.model.origin.grad.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/grad_norms/right_param', self.model.right_param.grad.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/grad_norms/up_param', self.model.up_param.grad.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/grad_norms/scaling', self.model.scaling_param.grad.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/scaling', self.model.scaling_param.item(), self.num_iters_done)

    def before_training_hook(self):
        self.plot_mask()
        self.save_mask()
        self.plot_all_weights_histograms()
        self.write_config()

    def after_training_hook(self):
        if self.is_explicitly_stopped:
            self.delete_logs() # So tensorboard does not lag
        else:
            self.visualize_minimum(self.vis_train_dataloader, 'train')
            self.visualize_minimum(self.vis_test_dataloader, 'test')

    def delete_logs(self):
        shutil.rmtree(self.config.firelab.logs_path)
        self.writer.close()

    def compute_mask_scores(self, dataloader):
        start = time.time()

        pad = self.config.get('solution_vis.padding', 1)
        x_num_points = self.config.get('solution_vis.granularity.x', self.mask.shape[0])
        y_num_points = self.config.get('solution_vis.granularity.y', self.mask.shape[1])
        xs = np.linspace(-pad, self.mask.shape[0] + pad, x_num_points)
        ys = np.linspace(-pad, self.mask.shape[1] + pad, y_num_points)

        dummy_model = self.torch_model_builder().to(self.config.firelab.device_name)
        scores = [[self.compute_mask_score(x, y, dummy_model, dataloader) for y in ys] for x in xs]
        self.logger.info(f'Scoring took {time.time() - start}')

        return xs, ys, scores

    def compute_mask_score(self, x, y, dummy_model, dataloader):
        w = self.model.compute_point(x, y, should_orthogonalize=True)

        return validate_weights(w, dataloader, dummy_model)

    def visualize_minimum(self, dataloader:DataLoader, subtitle:str):
        xs, ys, scores = self.compute_mask_scores(dataloader)
        fig = self.build_minimum_figure(xs, ys, scores, subtitle)
        self.writer.add_figure(f'Minimum_{subtitle}', fig, self.num_iters_done)
        self.save_minima_grid(scores, subtitle)

    def build_minimum_figure(self, xs, ys, scores, subtitle:str):
        X, Y = np.meshgrid(xs, ys)

        fig = plt.figure(figsize=(20, 4))

        plt.subplot(141)
        cntr = plt.contourf(X, Y, [[s[0] for s in s_row] for s_row in scores], cmap="RdBu_r", levels=np.linspace(0.3, 2.5, 30))
        plt.title(f'Loss [{subtitle}]')
        plt.colorbar(cntr)

        plt.subplot(142)
        cntr = plt.contourf(X, Y, [[s[1] for s in s_row] for s_row in scores], cmap="RdBu_r", levels=np.linspace(0.5, 0.9, 30))
        plt.title(f'Accfuracy [{subtitle}]')
        plt.colorbar(cntr)

        plt.subplot(143)
        cntr = plt.contourf(X, Y, [[s[0] for s in s_row] for s_row in scores], cmap="RdBu_r", levels=100)
        plt.title(f'Loss [{subtitle}]')
        plt.colorbar(cntr)

        plt.subplot(144)
        cntr = plt.contourf(X, Y, [[s[1] for s in s_row] for s_row in scores], cmap="RdBu_r", levels=np.linspace(0, 1, 100))
        plt.title(f'Accfuracy [{subtitle}]')
        plt.colorbar(cntr)

        return fig

    def validate(self):
        self.model.is_good_mode = True
        good_val_loss, good_val_acc = validate(self.model, self.train_dataloader, self.criterion)
        self.model.is_good_mode = False
        bad_val_loss, bad_val_acc = validate(self.model, self.train_dataloader, self.criterion)
        self.model.is_good_mode = True

        self.writer.add_scalar('good/val/loss', good_val_loss, self.num_epochs_done)
        self.writer.add_scalar('good/val/acc', good_val_acc, self.num_epochs_done)
        self.writer.add_scalar('bad/val/loss', bad_val_loss, self.num_epochs_done)
        self.writer.add_scalar('bad/val/acc', bad_val_acc, self.num_epochs_done)
        self.writer.add_scalar('diff/val/loss', good_val_loss - bad_val_loss, self.num_epochs_done)
        self.writer.add_scalar('diff/val/acc', good_val_acc - bad_val_acc, self.num_epochs_done)

        self.plot_all_weights_histograms()

        if self.num_epochs_done > self.config.get('val_acc_stop_threshold_num_warmup_epochs', -1):
            if good_val_acc < self.config.get('good_val_acc_stop_threshold', 0.):
                self.stop(f'Good val accuracy is too low (epoch #{self.num_epochs_done}): {good_val_acc}')
            elif bad_val_acc > self.config.get('bad_val_acc_stop_threshold', 1.):
                self.stop(f'Bad val accuracy is too high (epoch #{self.num_epochs_done}): {bad_val_acc}')
            else:
                pass

        if self.num_epochs_done > self.config.get('diff_threshold_num_warmup_epochs', -1):
            if good_val_acc - bad_val_acc < self.config.get('good_and_bad_val_acc_diff_threshold', float('-inf')):
                self.stop(f'Difference between good and val accuracies is too small '\
                          f'(epoch #{self.num_epochs_done}): {good_val_acc} - {bad_val_acc} = {good_val_acc - bad_val_acc}')

    def plot_mask(self):
        fig = plt.figure(figsize=(5, 5))
        mask_img = np.copy(self.mask)
        mask_img[mask_img == 2] = 0.5
        plt.imshow(mask_img, cmap='gray')
        self.writer.add_figure('Mask', fig, self.num_iters_done)

    def save_mask(self):
        save_path = os.path.join(self.config.firelab.custom_data_path, 'mask.npy')
        np.save(save_path, self.mask)

    def plot_params_histograms(self, w, subtag:str):
        dummy_model = self.torch_model_builder()
        params = weight_to_param(w, param_sizes(dummy_model.parameters()))
        tags = ['Weights_histogram_{}/{}'.format(i, subtag) for i in range(len(params))]

        for tag, param in zip(tags, params):
            self.writer.add_histogram(tag, param, self.num_iters_done)

    def plot_all_weights_histograms(self):
        # TODO: we do not need histograms currently...
        # self.plot_params_histograms(self.model.origin + self.model.right, 'origin_right')
        # self.plot_params_histograms(self.model.origin + self.model.up, 'origin_up')
        # self.plot_params_histograms(self.model.origin + self.model.up + self.model.right, 'origin_up_right')
        pass

    def write_config(self):
        config_yml = yaml.safe_dump(self.config.to_dict())
        config_yml = config_yml.replace('\n', '  \n') # Because tensorboard uses markdown
        self.writer.add_text('Config', config_yml, self.num_iters_done)

    def save_minima_grid(self, scores, subtitle:str):
        save_path = os.path.join(self.config.firelab.custom_data_path, f'minima_grid_{subtitle}.npy')
        np.save(save_path, scores)


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
    useless_cells:List[Tuple[int, int]] = []

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            num_ones = mask[max(i-1, 0):i+2, max(j-1, 0):j+2].sum()

            if mask[i][j] == 0 and num_ones == 0 or \
               mask[i][j] == 1 and num_ones == 9 or \
               mask[i][j] == 1 and is_corner(i, j, mask.shape) and num_ones == 4 or \
               mask[i][j] == 1 and is_border(i, j, mask.shape) and num_ones == 6:
                useless_cells.append((i, j))

    result = np.copy(mask)
    result[[i for i,j in useless_cells], [j for i,j in useless_cells]] = 2

    # Convert 0/1/2 mask to -1/0/1 mask, because it's more sensible
    result[result == 0] = -1
    result[result == 2] = 0

    return result


def is_corner(i, j, shape):
    if i == 0 and j == 0: return True
    if i == 0 and j == shape[1]-1: return True
    if i == shape[0]-1 and j == 0: return True
    if i == shape[0]-1 and j == shape[1]-1: return True

    return False


def is_border(i, j, shape):
    if is_corner(i, j, shape): return False
    if i == 0 or j == 0: return True
    if i == shape[0]-1 or j == shape[1]-1: return True

    return False


def randomly_fill_square(mask:np.array, p:float=1) -> np.array:
    "Takes square mask and fills it"
    assert mask.shape[0] == mask.shape[1]
    assert mask.shape[0] >= 4

    result = np.copy(mask)
    interior = np.random.rand(*result[2:-2, 2:-2].shape) < p
    result[2:-2, 2:-2] = interior

    return result


def convert_img_to_binary(img, threshold:float=0.5):
    if img.ndim == 2:
        return (img > threshold).astype(np.float)
    else:
        return (img[:, :, 3] > threshold).astype(np.float)


def generate_square_grid_mask(n_good_cells:int) -> np.ndarray:
    mask = np.zeros((2 * n_good_cells + 1, 2 * n_good_cells + 1))
    pos_cells_x_idx, pos_cells_y_idx = np.meshgrid(np.arange(1, 2 * n_good_cells, 2), np.arange(1, 2 * n_good_cells, 2))
    mask[pos_cells_x_idx, pos_cells_y_idx] = 1

    return mask