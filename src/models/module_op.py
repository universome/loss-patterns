from typing import List
import torch.nn.functional as F

class ModuleOperation:
    def __init__(self):
        self.trainig = True

    def train(self):
        self.training = True

        for m in self.get_modules():
            m.train()

    def eval(self):
        self.training = False

        for m in self.get_modules():
            m.eval()

    def get_modules(self):
        return []

    # def to(self, *args, **kwargs):
    #     for name, p in self.named_params:
    #         setattr(self, name, p.to(*args, **kwargs))
