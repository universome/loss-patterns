"""
Exports data from tensorboard logs to csv for our experiments
"""
import os
import argparse
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm
from firelab.utils.fs_utils import load_config, clean_dir
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main(experiment_dir:os.PathLike, output_dir:os.PathLike):
    logs_dir = os.path.join(experiment_dir, 'logs')

    # TODO: use configs instead of summaries when the bug is fixed
    configs_dir = os.path.join(experiment_dir, 'summaries')

    hpo_exp_names = sorted([exp for exp in os.listdir(logs_dir)])
    hps = []
    val_acc_diffs = []
    finished_exps = []
    images = []

    for i, hpo_exp_name in tqdm(enumerate(hpo_exp_names), total=len(hpo_exp_names)):
        assert len(get_dir_children(os.path.join(logs_dir, hpo_exp_name))) == 1, \
            f"Multiple logs are stored in {os.path.join(logs_dir, hpo_exp_name)}. That's ambigous."

        # if i > 1: break

        config_path = os.path.join(configs_dir, f'{hpo_exp_name}.yml')
        logs_path = get_dir_children(os.path.join(logs_dir, hpo_exp_name))[0]

        if not os.path.exists(config_path):
            print(f'Skipping {hpo_exp_name} because summary does not exist yet')
            continue

        curr_val_acc_diffs, curr_hp, curr_image = extract_data(config_path, logs_path)

        if len(curr_val_acc_diffs) != 100:
            print(f'Skipping {hpo_exp_name} because not finished yet (only {len(curr_val_acc_diffs)})')
            continue

        if curr_image is None:
            print(f'Skipping {hpo_exp_name} because image does not yet')
            continue

        val_acc_diffs.append(curr_val_acc_diffs)
        hps.append(curr_hp)
        images.append(curr_image)
        finished_exps.append(hpo_exp_name)

    # print('val_acc_diffs', val_acc_diffs)
    # print('hps', hps)
    # print('finished_exps', finished_exps)

    val_acc_diffs = pd.DataFrame.from_dict({exp: values for exp, values in zip(finished_exps, val_acc_diffs)})
    hps = pd.DataFrame(data=hps, index=finished_exps)

    clean_dir(output_dir, create=True)
    val_acc_diffs.to_csv(os.path.join(output_dir, 'val_acc_diffs.csv'))
    hps.to_csv(os.path.join(output_dir, 'hps.csv'))

    os.mkdir(os.path.join(output_dir, 'images'))
    for exp_name, image in zip(finished_exps, images):
        with open(os.path.join(output_dir, 'images', f'{exp_name}.png'), 'wb') as f:
            f.write(image)


def extract_data(config_path:os.PathLike, logs_path:os.PathLike) -> Tuple[List[float], Dict, str]:
    # TODO: we load summary and not config here. Use config after the bug is fixed
    config = load_config(config_path).config
    events_acc = EventAccumulator(logs_path)
    events_acc.Reload()

    _, _, val_acc_diffs = zip(*events_acc.Scalars('diff/val/acc'))
    hp = config.hp.to_dict()
    hp['n_conv_layers'] = len(config.hp.conv_model_config.conv_sizes)

    if 'Minimum' in events_acc.images.Keys():
        image = events_acc.Images('Minimum')[0].encoded_image_string
    else:
        image = None

    return val_acc_diffs, hp, image



def get_dir_children(dir:str) -> List[str]:
    return [os.path.join(dir, f) for f in os.listdir(dir)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Expoert tensorboard experimentcsv')
    parser.add_argument('-e', '--experiment_dir', type=str, required=True, metavar='experiment_dir',
        help='Path to experiment directory (for example, "experiments/super-experiment-00017/experiment"')
    parser.add_argument('-o', '--output_dir', type=str, required=True, metavar='output_dir',
        help='Path to directory where to save the results')

    args = parser.parse_args()

    main(args.experiment_dir, args.output_dir)
