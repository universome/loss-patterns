import os

from firelab import BaseTrainer
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from firelab import BaseTrainer
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils import weight_vector, validate, linerp, elbow_interpolation_scores
from src.models import ElbowModel, SimpleModel
from src.trainers.classifier_trainer import ClassifierTrainer


class ElbowTrainerWrapper:
    def __init__(self, config):
        self.config = config
        self.trainer_a = ClassifierTrainer(config)
        self.trainer_b = ClassifierTrainer(config)
        self.elbow_trainer = ElbowTrainer(config)

    def start(self):
        self.trainer_a.start()
        self.trainer_b.start()

        w_1 = weight_vector(self.trainer_a.model.parameters())
        w_2 = weight_vector(self.trainer_b.model.parameters())

        self.elbow_trainer.set_weights(w_1, w_2)
        self.elbow_trainer.start()


class ElbowTrainer(BaseTrainer):
    def __init__(self, config):
        super(ElbowTrainer, self).__init__(config)

        self.w_1 = None
        self.w_2 = None

    def init_models(self):
        self.model = ElbowModel(w_1=self.w_1, w_2=self.w_2).to(self.config.firelab.device_name)

    def set_weights(self, w_1, w_2):
        self.w_1 = w_1
        self.w_2 = w_2

    def before_start_hook(self):
        assert not (self.w_1 is None or self.w_2 is None)

    def init_dataloaders(self):
        batch_size = self.config.hp.batch_size
        project_path = self.config.firelab.project_path
        data_dir = os.path.join(project_path, self.config.data_dir)

        data_train = FashionMNIST(data_dir, download=True, train=True, transform=ToTensor())
        data_test = FashionMNIST(data_dir, download=True, train=False, transform=ToTensor())

        self.train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=3, shuffle=True)
        self.val_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=3, shuffle=False)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def init_optimizers(self):
        self.optim = Adam([self.model.w_3], lr=self.config.hp.lr)

    def train_on_batch(self, batch):
        self.model.train()

        x = batch[0].to(self.config.firelab.device_name)
        y = batch[1].to(self.config.firelab.device_name)

        preds = self.model(x)
        loss = self.criterion(preds, y).mean()
        acc = (preds.argmax(dim=1) == y).float().mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Train/loss', loss.item(), self.num_iters_done)
        self.writer.add_scalar('Train/acc', acc.item(), self.num_iters_done)
        self.writer.add_scalar('Stats/w_3_norm', self.model.w_3.norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/dist_to_w_1', (self.model.w_1 - self.model.w_3).norm().item(), self.num_iters_done)
        self.writer.add_scalar('Stats/dist_to_w_2', (self.model.w_2 - self.model.w_3).norm().item(), self.num_iters_done)

    def validate(self):
        self.model.eval()
        loss, acc = validate(self.model, self.val_dataloader, self.criterion)

        self.writer.add_scalar('Val/loss', loss.item(), self.num_iters_done)
        self.writer.add_scalar('Val/acc', acc.item(), self.num_iters_done)

    def visualize_interpolations(self):
        dummy_model = SimpleModel().to(self.config.firelab.device_name)

        w1to2_linerp_scores_train = linerp(self.w_1, self.w_2, dummy_model, self.train_dataloader)
        w1to2_linerp_scores_test = linerp(self.w_1, self.w_2, dummy_model, self.val_dataloader)
        elbow_linerp_scores_train = elbow_interpolation_scores(self.w_1, self.w_2, self.model.w_3, dummy_model, self.train_dataloader)
        elbow_linerp_scores_test = elbow_interpolation_scores(self.w_1, self.w_2, self.model.w_3, dummy_model, self.val_dataloader)

        self.writer.add_figure('w_1 to w_2 interpolation/accuracy', generate_interpolation_plot(
            [s[1] for s in w1to2_linerp_scores_train],
            [s[1] for s in w1to2_linerp_scores_test],
        ))

        self.writer.add_figure('w_1 to w_2 interpolation/loss', generate_interpolation_plot(
            [s[0] for s in w1to2_linerp_scores_train],
            [s[0] for s in w1to2_linerp_scores_test],
        ))

        self.writer.add_figure('elbow interpolation/accuracy', generate_interpolation_plot(
            [s[1] for s in elbow_linerp_scores_train],
            [s[1] for s in elbow_linerp_scores_test],
        ))

        self.writer.add_figure('elbow interpolation/loss', generate_interpolation_plot(
            [s[0] for s in elbow_linerp_scores_train],
            [s[0] for s in elbow_linerp_scores_test],
        ))

    def on_training_done(self):
        self.visualize_interpolations()


def generate_interpolation_plot(linerp_vals_train, linerp_vals_test, title:str=''):
    xs_train = np.linspace(0, 1, len(linerp_vals_train))
    xs_test = np.linspace(0, 1, len(linerp_vals_test))

    fig = plt.figure(figsize=(8, 5))
    if title != '': plt.title(title)
    plt.plot(xs_train, linerp_vals_train, label='Train')
    plt.plot(xs_test, linerp_vals_test, label='Test')
    plt.legend()
    plt.xlabel('alpha')
    plt.grid()

    return fig

