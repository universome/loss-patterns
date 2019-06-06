from typing import List
import torch.nn.functional as F

class ModuleOperation:
    def __init__(self):
        self.trainig = True

    def train(self, is_enabled:bool=True):
        self.training = is_enabled

        for m in self.get_modules():
            m.train(is_enabled)

        return self

    def eval(self):
        return self.train(False)

    def get_modules(self):
        return []

    # def to(self, *args, **kwargs):
    #     for name, p in self.named_params:
    #         setattr(self, name, p.to(*args, **kwargs))
