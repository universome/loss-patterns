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

from src.models import ElbowModel, SimpleModel
from src.trainers.classifier_trainer import ClassifierTrainer
from src.utils import weight_vector, validate, linerp, elbow_linerp_scores, get_weights_linerp
from src.utils import compute_weights_entropy_linerp, compute_activations_entropy_linerp
from src.plotting_utils import generate_linerp_plot, generate_acts_entropy_linerp_plot, generate_weights_entropy_linerp_plot


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

    def visualize_linerps(self):
        dummy_model = SimpleModel().to(self.config.firelab.device_name)

        w1to2_linerp_scores_train = linerp(self.w_1, self.w_2, dummy_model, self.train_dataloader)
        w1to2_linerp_scores_test = linerp(self.w_1, self.w_2, dummy_model, self.val_dataloader)
        elbow_linerp_scores_train = elbow_linerp_scores(self.w_1, self.w_2, self.model.w_3, dummy_model, self.train_dataloader)
        elbow_linerp_scores_test = elbow_linerp_scores(self.w_1, self.w_2, self.model.w_3, dummy_model, self.val_dataloader)

        self.writer.add_figure('w_1 to w_2 linerp/accuracy', generate_linerp_plot(
            [s[1] for s in w1to2_linerp_scores_train],
            [s[1] for s in w1to2_linerp_scores_test],
        ))

        self.writer.add_figure('w_1 to w_2 linerp/loss', generate_linerp_plot(
            [s[0] for s in w1to2_linerp_scores_train],
            [s[0] for s in w1to2_linerp_scores_test],
        ))

        self.writer.add_figure('elbow linerp/accuracy', generate_linerp_plot(
            [s[1] for s in elbow_linerp_scores_train],
            [s[1] for s in elbow_linerp_scores_test],
        ))

        self.writer.add_figure('elbow linerp/loss', generate_linerp_plot(
            [s[0] for s in elbow_linerp_scores_train],
            [s[0] for s in elbow_linerp_scores_test],
        ))

    def visualize_entropy_linerps(self):
        # We are visualizing only between w_1 and w_2 here
        dummy_model = SimpleModel().to(self.config.firelab.device_name)

        # acts_ent_linerp_train = compute_activations_entropy_linerp(self.w_1, self.w_2, dummy_model.nn, self.train_dataloader)
        acts_ent_linerp_test = compute_activations_entropy_linerp(self.w_1, self.w_2, dummy_model.nn, self.val_dataloader)
        weights_ent_linerp = compute_weights_entropy_linerp(self.w_1, self.w_2)

        # self.writer.add_figure('activations_entropy_linerp/train', generate_acts_entropy_linerp_plot(acts_ent_linerp_train))
        self.writer.add_figure('activations_entropy_linerp/test', generate_acts_entropy_linerp_plot(acts_ent_linerp_test))
        self.writer.add_figure('weights_entropy_linerp', generate_weights_entropy_linerp_plot(weights_ent_linerp))

        for i, w in enumerate(get_weights_linerp(self.w_1, self.w_2)):
            self.writer.add_histogram('Weights histogram', w.detach().cpu(), i)

    def after_training_hook(self):
        self.visualize_linerps()
        self.visualize_entropy_linerps()
