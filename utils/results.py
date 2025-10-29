import abc
import collections.abc

import pandas as pd
import matplotlib.pyplot as plt
import os

import numpy as np
import torch


def print_results(results_dict, relations_labels):

    output_msg = f'Training time (s): {results_dict["training_time"]:.3f}\n'
    for i, rel in enumerate(relations_labels):
        output_msg += '------------------------------------------------\n'
        output_msg += f'REL {i} ({rel}) RESULTS:\n'
        output_msg += f'Test accuracy rel {i}: {results_dict[f"accuracy_rel_{rel}"]:.3f}\n'
        output_msg += f'Test recall_0 rel {i}: {results_dict[f"recall_0_rel_{rel}"]:.3f}\n'
        output_msg += f'Test precision_0 rel {i}: {results_dict[f"precision_0_rel_{rel}"]:.3f}\n'
        output_msg += f'Test f1_score_0 rel {i}: {results_dict[f"f1_score_0_rel_{rel}"]:.3f}\n'
        output_msg += f'Test recall_1 rel {i}: {results_dict[f"recall_1_rel_{rel}"]:.3f}\n'
        output_msg += f'Test precision_1 rel {i}: {results_dict[f"precision_1_rel_{rel}"]:.3f}\n'
        output_msg += f'Test f1_score_1 rel {i}: {results_dict[f"f1_score_1_rel_{rel}"]:.3f}\n'
        output_msg += f'Test auc rel {i}: {results_dict[f"auc_rel_{rel}"]:.3f}\n'
        output_msg += f'True negative rel {i}: {results_dict[f"auc_rel_{rel}"]:.3f}\n'
        output_msg += f'Test auc rel {i}: {results_dict[f"auc_rel_{rel}"]:.3f}\n'
        output_msg += f'True negative rel {i}: {results_dict[f"tn_rel_{rel}"]:.3f}\n'
        output_msg += f'True positive rel {i}: {results_dict[f"tp_rel_{rel}"]:.3f}\n'
        output_msg += f'False negative rel {i}: {results_dict[f"fn_rel_{rel}"]:.3f}\n'
        output_msg += f'False positive rel {i}: {results_dict[f"fp_rel_{rel}"]:.3f}\n'

    print(output_msg)

def get_config_name(name_params, config):
    if isinstance(config, collections.abc.Sequence):
        return '_'.join([name_params[idx] + ':' + str(config[idx]) for idx, _ in enumerate(name_params)])
    elif isinstance(config, collections.abc.Mapping):
        return '_'.join([name + ':' + str(config[name]) for idx, name in enumerate(name_params)])

def save_results(results_dict, dataset, model_name, name_params, config, optimizer='adam', name=None):

    if name:
        savepath = f'./results/{dataset}_{optimizer.lower()}_{name}/'
    else:
        savepath = f'./results/{dataset}_{optimizer.lower()}/'

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    results_df = pd.DataFrame([results_dict])
    config_name = get_config_name(name_params, config)
    results_df.to_csv(f'{savepath}test_{model_name.lower()}_{config_name}.tsv', sep='\t', index=None)


def save_results_best(results_dict, dataset, best_model, model_name, name_params, config, optimizer='adam', name=None):
    if name:
        savepath = f'./results/{dataset}_{optimizer.lower()}_{name}/'
    else:
        savepath = f'./results/{dataset}_{optimizer.lower()}/'

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    results_df = pd.DataFrame([results_dict])
    results_df['model'] = get_config_name(name_params, config)
    results_df.to_csv(f'{savepath}/test_{model_name.lower()}_best.tsv', sep='\t', index=None)

    torch.save(best_model, f"{savepath}/{model_name.lower()}_best_model.pth")


def save_losses_best(losses, dataset, model_name, optimizer='adam', name=None):
    # Convert losses to numpy arrays
    train_losses = np.array(losses['train'])
    val_losses = np.array(losses['val'])
    train_acc_losses = np.array(losses['train_acc'])
    val_acc_losses = np.array(losses['val_acc'])

    # Stack them column-wise and save
    loss_data = np.column_stack((train_losses, val_losses))
    acc_data = np.column_stack((train_acc_losses, val_acc_losses))

    if name:
        savepath = f'./results/{dataset}_{optimizer.lower()}_{name}/'
    else:
        savepath = f'./results/{dataset}_{optimizer.lower()}/'

    np.savetxt(f'{savepath}{model_name.lower()}_best_losses.txt', loss_data, header='train_loss\tval_loss')
    np.savetxt(f'{savepath}{model_name.lower()}_best_accs.txt', acc_data, header='train_acc\tval_acc')
