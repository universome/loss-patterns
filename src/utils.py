from collections import OrderedDict
from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm
from firelab.utils.training_utils import get_module_device


def weight_vector(params):
    return torch.cat([p.view(-1) for p in params])


def params_diff(p_lhs, p_rhs):
    return [lhs - rhs for lhs, rhs in zip(p_lhs, p_rhs)]


def model_from_weight(weight, model_builder):
    model = model_builder().to(weight.device)
    param = weight_to_param(weight, param_sizes(model.parameters()))
    state_dict = params_to_state_dict(param, model.state_dict().keys())
    model.load_state_dict(state_dict)

    return model


def params_sum(p_lhs, p_rhs):
    return [lhs + rhs for lhs, rhs in zip(p_lhs, p_rhs)]


def params_scalar_mult(params, x:float):
    return [p * x for p in params]


def params_dot_product(p_lhs, p_rhs):
    return sum([torch.dot(lhs.view(-1), rhs.view(-1)) for lhs, rhs in zip(p_lhs, p_rhs)])


def param_sizes(param):
    return [p.size() for p in param]


def weight_to_param(w, sizes):
    params = []

    for s in sizes:
        curr_block = w[:np.prod(s)]
        p = curr_block.view(s)
        w = w[np.prod(s):]
        params.append(p)

    return params


def orthogonalize(theta, r, adjust_len=False):
    theta = theta.double()
    r = r.double()
    #z_tilde = theta * torch.norm(r).pow(2) - r * torch.dot(theta, r)
    z = theta - r * (torch.dot(theta, r) / torch.norm(r).pow(2))

    if adjust_len:
        z = z / torch.norm(z) * torch.norm(r)

    return z.float()


def sample_on_circle(center, z, r, angle_range=(0, 2 * np.pi)):
    alpha = np.random.uniform(*angle_range)
    w = center + np.sin(alpha).item() * z + np.cos(alpha).item() * r

    return w


def sample_on_square(center, z, r):
    if np.random.rand() > 0.5:
        w = center + np.random.rand() * z + r
    else:
        w = center + z + np.random.rand() * r

    return w


def params_to_state_dict(params, keys):
    return OrderedDict([(k,p) for k,p in zip(keys, params)])


def validate(model, dataloader, criterion) -> Tuple[float, float]:
    model.eval()
    guessed = np.array([])
    losses = np.array([])
    device = get_module_device(model)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            losses = np.hstack([losses, loss.cpu().numpy()])
            guessed = np.hstack([guessed, (preds.argmax(dim=1) == y).long().cpu().numpy()])

    return losses.mean(), guessed.mean()


def validate_weights(weights, dataloader, model):
    # model = SuperModel().to(device)
    params = weight_to_param(weights, param_sizes(model.parameters()))
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.load_state_dict(params_to_state_dict(params, model.state_dict().keys()))

    return validate(model, dataloader, criterion)


def validate_random_viscinity(weights):
    raise NotImplementedError


def compute_entropy(values, num_bins:int=100):
    hist, _ = np.histogram(values, bins=num_bins)
    probs = hist / len(values)
    non_zero_probs = probs[probs != 0]
    entropy = (-non_zero_probs * np.log(non_zero_probs)).sum()

    return entropy


def get_activations_for_sequential(sequential_model, x):
    activations = [x]

    with torch.no_grad():
        for m in sequential_model.children():
            activations.append(m(activations[-1]))

    return [a.cpu().view(-1) for a in activations]


def compute_activations_entropy(model, dataloader, num_bins:int=100):
    act_history = []

    for x, y in dataloader:
        #act_hist.append(model.get_activations(x))
        act_history.append(get_activations_for_sequential(model.nn, x.to(device)))

    # Transposing array
    act_history = [[acts[i] for acts in act_history] for i in range(len(act_history[0]))]
    activations = [torch.cat(acts) for acts in act_history]
    entropies = [compute_entropy(acts, num_bins) for acts in activations]

    return entropies


def linerp(w_a, w_b, model, dataloader, n_steps:int=25):
    alphas = np.linspace(0, 1, n_steps)
    weights = [w_a * (1 - alpha) + w_b * alpha for alpha in alphas]
    val_scores = [validate_weights(w, dataloader, model=model) for w in weights]

    return val_scores


def elbow_interpolation_scores(w_a, w_b, w_c, model, dataloader, n_steps:int=25):
    a2c_scores = linerp(w_a, w_c, model, n_steps, dataloader)
    c2b_scores = linerp(w_c, w_b, model, n_steps, dataloader)

    return a2c_scores + c2b_scores
