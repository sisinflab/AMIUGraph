import time

import pandas as pd
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, confusion_matrix
from utils.dataset import shuffle_extract_batch_kg
from utils.determinism import enable_determinism
from abc import abstractmethod
from typing import List
import copy


class KgeModel(torch.nn.Module):
    def __init__(
        self,
        num_set_node,
        relations: List[str],
        conf,
        random_seed,
        optimizer='Adam',
        inductive_ids=None,
        inductive_embs=None,
        is_inductive=False,
    ):
        super().__init__()

        enable_determinism(random_seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = conf
        self.num_set_node = num_set_node
        self.relations = relations
        self.num_relations = len(relations)
        self.reg_term = conf.get("reg_term", 0.001)
        self.emb_size = conf.get("emb_size", 64)
        self.learning_rate = conf.get("learning_rate", 0.01)
        self.opt_name = optimizer
        self.inductive_ids = torch.tensor(inductive_ids, device=self.device) if inductive_ids is not None else torch.tensor([], device=self.device)
        if inductive_embs is not None:
            self.inductive_embs = torch.nn.Parameter(torch.tensor(inductive_embs, dtype=torch.float32, device=self.device))  # Make it a trainable parameter
            if not is_inductive:
                self.inductive_embs.requires_grad = False
        else:
            self.inductive_embs = []

        self.is_inductive = is_inductive


    def set_trainable_params(self, new_params, l2_decay=0):
        # Ensure new_params is a list (in case a single parameter is passed)
        if not isinstance(new_params, list):
            new_params = [new_params]

        # Check if the optimizer already exists
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            # Add new parameters to the existing optimizer
            for param in new_params:
                if param not in list(p['params'] for p in self.optimizer.param_groups):
                    self.optimizer.add_param_group({'params': param, 'weight_decay': l2_decay})
        else:
            # If no optimizer exists yet, create it
            if self.opt_name.lower() == 'adam':
                self.optimizer = torch.optim.Adam(new_params, lr=self.learning_rate, weight_decay=l2_decay)
            elif self.opt_name.lower() == 'adagrad':
                self.optimizer = torch.optim.Adagrad(new_params, lr=self.learning_rate, weight_decay=l2_decay)
            else:
                raise NotImplementedError("Optimizer not supported")

    @abstractmethod
    def lookup(self, inputs):
        pass


    @abstractmethod
    def lookup_ent(self, inputs):
        pass


    @abstractmethod
    def score(self, head, tail, rel) -> torch.Tensor:
        pass

    @abstractmethod
    def train_step_pre(self):
        pass

    @abstractmethod
    def train_step_post(self):
        pass

    @abstractmethod
    def loss(self, inputs, labels) -> float:
        pass

    def train_step(self, inputs):
        with torch.no_grad():
            self.train_step_pre()

        inputs_rel_idx = [inputs[2][inputs[2] == i].index.tolist() for i in range(self.num_relations)]

        labels = torch.tensor(inputs[3].tolist(), dtype=torch.float32, device=self.device)
        batch_loss, out = self.loss(inputs, labels)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.train_step_post()

        preds = out.reshape(-1).detach().cpu().numpy()
        preds_labels = (preds >= 0.5).astype(int)
        labels = labels.detach().cpu().numpy().astype(int)

        train_accuracies = [
            accuracy_score(y_true=labels[inputs_rel_idx[i]], y_pred=preds_labels[inputs_rel_idx[i]]) if len(
                labels[inputs_rel_idx[i]]) else 0.0 for i in range(self.num_relations)]

        return batch_loss.detach().cpu().numpy().item(), train_accuracies

    def fit(self,
            epochs,
            num_interactions,
            patience,
            train_edge_index_df,
            train_neg_edges,
            batch_size,
            val_edge_index_df,
            val_neg_edges):

        val_accuracies = []
        best_epoch = 1
        best_val_accuracy = -np.inf
        all_train_accuracies = []
        all_val_accuracies = []
        all_train_losses = []
        all_val_losses = []

        start_time = time.perf_counter()
        # train loop
        for ep in range(epochs):
            average_train_loss = 0
            average_val_loss = 0
            average_train_accuracies = [0] * self.num_relations
            average_val_accuracies = [0] * self.num_relations
            average_val_f1_scores = [0] * self.num_relations
            print('EPOCH {0}/{1}...\n'.format(ep + 1, epochs))
            with tqdm(total=num_interactions) as t:
                for batch in shuffle_extract_batch_kg(train_edge_index_df, train_neg_edges, batch_size):
                    current_loss_train, train_accuracies = self.train_step(batch)
                    val_dict = self.predict(val_edge_index_df, val_neg_edges)
                    average_train_loss += current_loss_train
                    average_train_accuracies = [sum(value) for value in zip(train_accuracies, average_train_accuracies)]
                    average_val_loss += val_dict['loss']
                    average_val_accuracies += [sum(value) for value in zip(val_dict['accuracy_rel'], average_val_accuracies)]
                    average_val_f1_scores += [sum(value) for value in zip(val_dict['f1_score_rel'], average_val_f1_scores)]

                    postfix = {'train_loss': f'{current_loss_train:.5f}', 'val_loss': f'{val_dict["loss"]:.5f}'}

                    for i in range(self.num_relations):
                        postfix.update({f'train_accuracy_rel_{i}': f'{train_accuracies[i]:.5f}',
                                        f'val_accuracy_rel_{i}': f'{val_dict["accuracy_rel"][i]:.5f}',
                                        f'val_f1_score_rel_{i}': f'{val_dict["f1_score_rel"][i]:.5f}'})
                    t.set_postfix(postfix)
                    t.update()

            all_train_losses += [average_train_loss / num_interactions]
            all_val_losses += [average_val_loss / num_interactions]
            val_accuracies.append((sum(average_val_accuracies) / len(average_val_accuracies)) / num_interactions)

            epoch_log = (f'\nEND EPOCH {ep + 1}.'
                         f'\tAvg train loss: {average_train_loss / num_interactions:.3f}'
                         f'\tAvg val loss: {average_val_loss / num_interactions:.3f}')

            for i in range(self.num_relations):
                epoch_log += f'\tAvg train accuracy rel {i}: {average_train_accuracies[i] / num_interactions:.3f}'
                epoch_log += f'\tAvg val accuracy rel {i}: {average_val_accuracies[i] / num_interactions:.3f}'
                epoch_log += f'\tAvg val f1 score rel {i}: {average_val_f1_scores[i] / num_interactions:.3f}'
                epoch_log += '\n'

            all_train_accuracies += [(sum(average_train_accuracies) / self.num_relations) / num_interactions]
            all_val_accuracies += [(sum(average_val_accuracies) / self.num_relations) / num_interactions]

            if best_val_accuracy < all_val_accuracies[-1]:
                best_epoch = ep + 1
                best_val_accuracy = all_val_accuracies[-1]
                best_model_weights = copy.deepcopy(self.state_dict())  # Deep copy the state dict

            if (ep + 1) >= patience:
                early_stopping = [True if val_accuracies[-1] < l else False for l in val_accuracies[:-1]]
                if all(early_stopping):
                    print('Met early stopping strategy!')
                    self.load_state_dict(best_model_weights)
                    break
                else:
                    val_accuracies = val_accuracies[1:]

        print('\nTRAINING ENDED!\n')
        # Calculate elapsed time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Training took: {elapsed_time:.3f}")

        print(f'Best epoch: {best_epoch}\t Best validation accuracy: {best_val_accuracy:.3f}')

        return best_val_accuracy, all_train_accuracies, all_val_accuracies, all_train_losses, all_val_losses, elapsed_time

    def predict(self, inputs_pos, inputs_neg, validation=True):

        inputs = pd.concat([pd.DataFrame(inputs_pos), pd.DataFrame(inputs_neg)], axis=0).reset_index(drop=True)
        inputs_rel_idx = [inputs[2][inputs[2] == i].index.tolist() for i in range(self.num_relations)]

        labels = torch.cat([torch.ones(inputs_pos.shape[0]), torch.zeros(inputs_neg.shape[0])])
        batch_loss, out = self.loss(inputs, labels)

        preds = out.reshape(-1).detach().cpu().numpy()
        preds_labels = (preds >= 0.5).astype(int)
        labels = labels.detach().cpu().numpy().astype(int)

        if validation:
            return {
                'loss': batch_loss.detach().cpu().numpy().item(),
                'accuracy_rel': [accuracy_score(y_true=labels[rel], y_pred=preds_labels[rel]) for rel in
                                 inputs_rel_idx],
                'f1_score_rel': [f1_score(y_true=labels[rel], y_pred=preds_labels[rel], zero_division=0) for rel in
                                 inputs_rel_idx],
            }
        else:

            results_rel = [classification_report(y_true=labels[rel], y_pred=preds_labels[rel], output_dict=True,
                                                 zero_division=0) for rel in inputs_rel_idx]

            confusion_matrix_rel = [dict(zip(['tn', 'fp', 'fn', 'tp'], confusion_matrix(y_true=labels[rel], y_pred=preds_labels[rel]).ravel()))
                                    for rel in inputs_rel_idx]

            result = {}


            for i, rel in enumerate(self.relations):
                result.update({
                    f'accuracy_rel_{rel}': results_rel[i]['accuracy'],
                    f'precision_0_rel_{rel}': results_rel[i]['0']['precision'],
                    f'recall_0_rel_{rel}': results_rel[i]['0']['recall'],
                    f'f1_score_0_rel_{rel}': results_rel[i]['0']['f1-score'],
                    f'precision_1_rel_{rel}': results_rel[i]['1']['precision'],
                    f'recall_1_rel_{rel}': results_rel[i]['1']['recall'],
                    f'f1_score_1_rel_{rel}': results_rel[i]['1']['f1-score'],
                    f'auc_rel_{rel}': roc_auc_score(y_true=labels[inputs_rel_idx[i]], y_score=preds[inputs_rel_idx[i]]),
                    f'tn_rel_{rel}': confusion_matrix_rel[i]['tn'],
                    f'fp_rel_{rel}': confusion_matrix_rel[i]['fp'],
                    f'fn_rel_{rel}': confusion_matrix_rel[i]['fn'],
                    f'tp_rel_{rel}': confusion_matrix_rel[i]['tp'],
                })

            return result
