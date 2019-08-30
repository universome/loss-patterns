import os
import shutil
import time
import random
from itertools import chain
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from firelab import BaseTrainer

from src.models.ensemble import MappingEnsemble, PlaneEnsemble
from src.utils import weight_vector
from src.trainers.mask_trainer import MaskTrainer


class EnsembleTrainer(BaseTrainer):
    def __init__(self, config):
        super(EnsembleTrainer, self).__init__(config)

    def init_models(self):
        MaskTrainer.init_torch_model_builder(self)

        if self.config.hp.ensemble_type == 'plane':
            self.model = PlaneEnsemble(
                self.torch_model_builder,
                self.config.hp.num_models,
                self.config.hp.coords_init_strategy
            )
        elif self.config.hp.ensemble_type == 'mapping':
            self.model = MappingEnsemble(
                self.torch_model_builder,
                self.config.hp.num_models,
                self.config.hp.ensemble_config
            )
        else:
            raise NotImplementedError(f'Unknown ensemble type: {self.config.hp.ensemble_type}')

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

            point_losses.append(loss)
            point_accs.append((preds.argmax(dim=1) == y).float().mean().item())
            point_preds.append(preds)

        total_loss = torch.stack(point_losses).mean()
        total_loss.backward(retain_graph=True)

        self.writer.add_scalar('Train/loss/mean', torch.stack(point_losses).mean().item(), self.num_iters_done)
        self.writer.add_scalar('Train/loss/max', torch.stack(point_losses).max().item(), self.num_iters_done)
        self.writer.add_scalar('Train/loss/min', torch.stack(point_losses).min().item(), self.num_iters_done)

        self.writer.add_scalar('Train/acc/mean', np.mean(point_accs), self.num_iters_done)
        self.writer.add_scalar('Train/acc/max', np.max(point_accs), self.num_iters_done)
        self.writer.add_scalar('Train/acc/min', np.min(point_accs), self.num_iters_done)

        # Adding ensemble decorrelation regularization
        # TODO:
        # - KL, MSE, other divergences on model outputs
        # - adversarial loss to guess from which model prediction has came
        # - just make weights farther from each other
        # - correlation/MSE for losses between models (yes, it will be differentiable)
        decorrelation_loss = self.compute_decorrelation(point_preds, y)

        if not decorrelation_loss is None:
            self.writer.add_scalar('Train/decorrelation', decorrelation_loss.item(), self.num_iters_done)
            decorrelation_loss *= self.config.hp.get('decorrelation_coef', 1.)
            decorrelation_loss.backward()

        self.writer.add_scalar('Stats/grad_norms/coords', self.model.coords.grad.norm().item(), self.num_iters_done)

        self.log_weight_stats()
        clip_grad_norm_(self.model.parameters(), self.config.hp.grad_clip_threshold)
        self.optim.step()

        self.scheduler.step()
        self.writer.add_scalar('Stats/lr', self.scheduler.get_lr()[0], self.num_iters_done)

    def log_weight_stats(self):
        if self.config.hp.ensemble_type == 'plane':
            # Weight l2 norms
            self.writer.add_scalar('Stats/norms/origin_param', self.model.origin_param.norm().item(), self.num_iters_done)
            self.writer.add_scalar('Stats/norms/right_param', self.model.right_param.norm().item(), self.num_iters_done)
            self.writer.add_scalar('Stats/norms/up_param', self.model.up_param.norm().item(), self.num_iters_done)

            # Grad norms
            self.writer.add_scalar('Stats/grad_norms/origin_param', self.model.origin_param.grad.norm().item(), self.num_iters_done)
            self.writer.add_scalar('Stats/grad_norms/right_param', self.model.right_param.grad.norm().item(), self.num_iters_done)
            self.writer.add_scalar('Stats/grad_norms/up_param', self.model.up_param.grad.norm().item(), self.num_iters_done)
        elif self.config.hp.ensemble_type == 'mapping':
            mapping_weight_norm = weight_vector(self.model.mapping.parameters()).norm()
            mapping_grad_norm = torch.cat([p.grad.view(-1) for p in self.model.mapping.parameters()]).norm()

            self.writer.add_scalar('Stats/norms/mapping', mapping_weight_norm.item(), self.num_iters_done)
            self.writer.add_scalar('Stats/grad_norms/mapping', mapping_grad_norm.item(), self.num_iters_done)
        else:
            pass

        self.writer.add_histogram('Coords/x', self.model.coords[:,0].cpu().detach().numpy(), self.num_iters_done)
        self.writer.add_histogram('Coords/y', self.model.coords[:,1].cpu().detach().numpy(), self.num_iters_done)

    def compute_decorrelation(self, preds:List[torch.Tensor], target:torch.Tensor):
        "Computes decorrelation regularization based on predictions"
        if self.config.hp.decorrelation_type == 'preds_distance':
            wrong_cls_mask = torch.ones_like(preds[0]).bool()
            wrong_cls_mask[torch.arange(preds[0].size(0)), target] = False
            wrong_cls_preds = torch.stack([p[wrong_cls_mask] for p in preds])
            pairs = [(i,j) for i in range(10) for j in range(i)]
            distances = [(wrong_cls_preds[p[0]] - wrong_cls_preds[p[1]]).pow(2).sum() for p in pairs]
            decorrelation_loss = torch.stack(distances).mean()
            decorrelation_loss = decorrelation_loss.clamp(0, 10) # Because model can just separate models apart and be happy
            decorrelation_loss *= -1 # Because we want the distance to be larger
        elif self.config.hp.decorrelation_type == 'weights_distance':
            ws = [self.model.get_model_weights_by_id(i) for i in range(self.model.coords.size(0))]
            pairs = [(i,j) for i in range(10) for j in range(i)]
            distances = [(ws[p[0]] - ws[p[1]]).pow(2).sum() for p in pairs]
            decorrelation_loss = torch.stack(distances).mean()
            decorrelation_loss = decorrelation_loss.clamp(0, 10) # Because model can just separate models apart and be happy
            decorrelation_loss *= -1 # Because we want the distance to be larger
        elif self.config.hp.decorrelation_type == 'none':
            decorrelation_loss = None
        else:
            raise NotImplementedError

        return decorrelation_loss

    def validate(self):
        individual_guessed = []
        individual_losses = []
        ensemble_guessed = []
        ensemble_losses = []

        self.model.eval()

        with torch.no_grad():
            for x, y in self.val_dataloader:
                x, y = x.to(self.device_name), y.to(self.device_name)
                individual_preds = torch.stack([self.model.run_model_by_id(i, x) for i in range(self.model.coords.size(0))])
                ensemble_preds = individual_preds.mean(dim=0)

                individual_losses.extend([self.criterion(p, y).cpu().mean().item() for p in individual_preds])
                ensemble_losses.append(self.criterion(ensemble_preds, y).cpu().mean().item())
                individual_guessed.extend([g for p in individual_preds for g in (p.argmax(dim=1) == y).cpu().numpy()])
                ensemble_guessed.extend([g for g in (ensemble_preds.argmax(dim=1) == y).cpu().numpy()])

        self.writer.add_scalar('Val/loss_diff', np.mean(ensemble_losses) - np.mean(individual_losses), self.num_epochs_done)
        self.writer.add_scalar('Val/acc_diff', np.mean(ensemble_guessed) - np.mean(individual_guessed), self.num_epochs_done)
        self.writer.add_scalar('Val/individual_mean_loss', np.mean(individual_losses), self.num_epochs_done)
        self.writer.add_scalar('Val/individual_mean_acc', np.mean(individual_guessed), self.num_epochs_done)
        self.writer.add_scalar('Val/ensemble_loss', np.mean(ensemble_losses), self.num_epochs_done)
        self.writer.add_scalar('Val/ensemble_acc', np.mean(ensemble_guessed), self.num_epochs_done)
