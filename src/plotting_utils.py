import numpy as np
import matplotlib.pyplot as plt

def generate_linerp_plot(linerp_vals_train, linerp_vals_test, title:str=''):
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


def generate_acts_entropy_linerp_plot(linerp_values):
    linerp_values = np.array(linerp_values)
    xs = np.linspace(0, 1, linerp_values.shape[1])
    colors = plt.cm.jet(np.linspace(0, 1, linerp_values.shape[0]))

    fig = plt.figure(figsize=(12, 7))
    for i, layer_entropies in enumerate(linerp_values):
        plt.plot(xs, layer_entropies, label='Layer #%d' % i, color=colors[i])
    plt.legend()
    plt.xlabel('alpha')
    plt.grid()

    return fig


def generate_weights_entropy_linerp_plot(values):
    xs = np.linspace(0, 1, len(values))

    fig = plt.figure(figsize=(7, 5))
    plt.plot(xs, values)
    plt.xlabel('alpha')
    plt.grid()

    return fig
