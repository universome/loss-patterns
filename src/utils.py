from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from firelab.utils.training_utils import get_module_device


def weight_vector(params):
    return torch.cat([p.view(-1) for p in params])


def params_diff(p_lhs, p_rhs):
    return [lhs - rhs for lhs, rhs in zip(p_lhs, p_rhs)]


def model_from_weight(weight, torch_model_cls):
    model = torch_model_cls().to(weight.device)
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


def weight_to_param(w, sizes) -> List[torch.Tensor]:
    params = []

    for s in sizes:
        curr_block = w[:np.prod(s)]
        p = curr_block.view(s)
        w = w[np.prod(s):]
        params.append(p)

    return params


def orthogonalize(v1, v2, adjust_len_to_v1=False, adjust_len_to_v2=False):
    """
    Performs Gram-Schmidt orthogonalization. Returns vector, orthogonal to v1
    If adjust_len_to_v1 (or adjust_len_to_v2) is provided,
    then it will have the same norm as v1 (or v2)

    TODO: add tests
    """
    assert not (adjust_len_to_v1 and adjust_len_to_v2), \
        "Impossible to adjust length to two vectors at the same time"

    v1 = v1.double()
    v2 = v2.double()

    #w = v1 * torch.norm(v2).pow(2) - v2 * torch.dot(v1, v2)
    result = v2 - v1 * (torch.dot(v2, v1) / torch.norm(v1).pow(2))

    if adjust_len_to_v1:
        result = result * (torch.norm(v1) / torch.norm(result))
    elif adjust_len_to_v2:
        result = result * (torch.norm(v2) / torch.norm(result))

    return result.float()


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
    state_dict = OrderedDict([])

    for key in keys:
        if 'running_mean' in key:
            state_dict[key] = torch.zeros_like(list(state_dict.values())[-1])
        elif 'running_var' in key:
            state_dict[key] = torch.ones_like(list(state_dict.values())[-1])
        elif 'num_batches_tracked' in key:
            state_dict[key] = torch.zeros(1, device=params[0].device)
        else:
            state_dict[key] = params[0]
            params = params[1:]

    return state_dict


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
    params = weight_to_param(weights, param_sizes(model.parameters()))
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.load_state_dict(params_to_state_dict(params, model.state_dict().keys()))

    return validate(model, dataloader, criterion)


def validate_random_viscinity(weights):
    raise NotImplementedError


def compute_entropy(values, num_bins:int=100):
    assert np.array(values).ndim == 1

    hist, _ = np.histogram(values, bins=num_bins)
    probs = hist / len(values)
    non_zero_probs = probs[probs != 0]
    entropy = (-non_zero_probs * np.log(non_zero_probs)).sum()

    return entropy


def get_activations_for_sequential(sequential_model, x):
    # It is useful to keep x for analysis and sanity check
    activations = [x]

    with torch.no_grad():
        for m in sequential_model.children():
            activations.append(m(activations[-1]))

    return [a.cpu().view(-1) for a in activations]


def compute_activations_entropy(model, dataloader, num_bins:int=100):
    num_layers = len(model)
    activations = [torch.Tensor([]) for _ in range(num_layers + 1)]
    device = get_module_device(model)

    for x, _ in dataloader:
        for i, act in enumerate(get_activations_for_sequential(model, x.to(device))):
            activations[i] = torch.cat([activations[i], act])

    entropies = [compute_entropy(acts, num_bins) for acts in activations]

    return entropies


def compute_activations_entropy_for_weights(w, model, dataloader, num_bins:int=100):
    params = weight_to_param(w, param_sizes(model.parameters()))
    model.load_state_dict(params_to_state_dict(params, model.state_dict().keys()))

    return compute_activations_entropy(model, dataloader, num_bins)


def compute_activations_entropy_linerp(w_1, w_2, model, dataloader, n_steps:int=25, num_bins:int=100):
    weights = get_weights_linerp(w_1, w_2, n_steps)
    entropies = [compute_activations_entropy_for_weights(w, model, dataloader, num_bins) for w in tqdm(weights)]
    entropies = np.array(entropies).transpose()

    return entropies


def compute_weights_entropy_linerp(w_1, w_2, n_steps:int=25, n_bins:int=1000):
    weights = get_weights_linerp(w_1, w_2, n_steps)
    weights = [w.detach().cpu() for w in weights]
    entropies = [compute_entropy(w, n_bins) for w in tqdm(weights)]

    return entropies


def get_weights_linerp(w_1, w_2, n_steps:int=25):
    alphas = np.linspace(0, 1, n_steps)
    weights = [w_1 * (1 - alpha) + w_2 * alpha for alpha in alphas]

    return weights


def linerp(w_1, w_2, model, dataloader, n_steps:int=25):
    weights = get_weights_linerp(w_1, w_2, n_steps)
    val_scores = [validate_weights(w, dataloader, model=model) for w in tqdm(weights)]

    return val_scores


def elbow_linerp_scores(w_1, w_2, w_elbow, model, dataloader, n_steps:int=25):
    a2c_scores = linerp(w_1, w_elbow, model, dataloader, n_steps=n_steps)
    c2b_scores = linerp(w_elbow, w_2, model, dataloader, n_steps=n_steps)

    return a2c_scores + c2b_scores
