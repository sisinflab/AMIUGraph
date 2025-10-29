import torch
import abc
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, confusion_matrix
from utils.dataset import shuffle_extract_batch
from utils.determinism import enable_determinism
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNNModel(torch.nn.Module):
    def __init__(self,
                 num_set_node_one,
                 num_set_node_two,
                 adj,
                 conf,
                 random_seed,
                 emb_set_node_one=None,
                 emb_set_node_two=None,
                 adj_extended=None):

        super().__init__()

        enable_determinism(random_seed)

        self.num_set_node_one = num_set_node_one
        self.num_set_node_two = num_set_node_two
        self.adj = adj
        self.adj_extended = adj_extended
        if adj_extended is None:
            self.adj_extended = self.adj

        self.reg_term = conf["reg_term"]
        self.emb_size = conf["emb_size"]
        self.learning_rate = conf["learning_rate"]
        self.projection_node_one = self.projection_node_two = None
        self.freeze_node_one = conf.get("freeze_node_one", False)
        self.freeze_node_two = conf.get("freeze_node_two", False)

        if emb_set_node_one is not None:
            self.emb_set_node_one = torch.nn.Parameter(emb_set_node_one.to(torch.float32))
            if self.emb_set_node_one.shape[-1] != self.emb_size:
                self.projection_node_one = torch.nn.Linear(in_features=self.emb_set_node_one.shape[-1],
                                                           out_features=self.emb_size).to(device)
        else:
            self.emb_set_node_one = torch.nn.Parameter(torch.nn.init.xavier_normal_(
                torch.empty((self.num_set_node_one, self.emb_size))))

        if emb_set_node_two is not None:
            self.emb_set_node_two = torch.nn.Parameter(emb_set_node_two.to(torch.float32))
            if self.emb_set_node_two.shape[-1] != self.emb_size:
                self.projection_node_two = torch.nn.Linear(in_features=self.emb_set_node_two.shape[-1],
                                                           out_features=self.emb_size).to(device)
        else:
            self.emb_set_node_two = torch.nn.Parameter(torch.nn.init.xavier_normal_(
                torch.empty((self.num_set_node_two, self.emb_size))))

        self.emb_set_node_one.requires_grad = not self.freeze_node_one
        self.emb_set_node_two.requires_grad = not self.freeze_node_two

        self.emb_set_node_one.to(device)
        self.emb_set_node_two.to(device)
        self.adj.to(device)
        self.adj_extended.to(device)
        self.optimizer = None


    def lookup_ent(self, inputs):
        emb_set_node_one_update, emb_set_node_two_update = self.propagate_embeddings()
        return emb_set_node_one_update[inputs]
       

    @abc.abstractmethod
    def get_trainable_parameters(self):
        pass

    @abc.abstractmethod
    def _propagate_embeddings(self, all_node_embeddings, adj):
        pass

    @abc.abstractmethod
    def train_step(self, inputs):
        pass

    def propagate_embeddings(self, extend_adj=False):
        if extend_adj:
            adj = self.adj_extended.to(device)
        else:
            adj = self.adj.to(device)

        projected_node_one = self.emb_set_node_one
        projected_node_two = self.emb_set_node_two
        if self.projection_node_one:
            projected_node_one = self.projection_node_one(self.emb_set_node_one.to(device))
        if self.projection_node_two:
            projected_node_two = self.projection_node_two(self.emb_set_node_two.to(device))
        all_node_embeddings = torch.cat([projected_node_one.to(device), projected_node_two.to(device)], dim=0)
        all_node_embeddings = torch.nn.functional.normalize(all_node_embeddings, p=2, dim=-1)
        all_node_embeddings.to(device)

        return self._propagate_embeddings(all_node_embeddings, adj)


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

        self.optimizer = torch.optim.Adam(self.get_trainable_parameters(), lr=self.learning_rate)

        start_time = time.perf_counter()
        for ep in range(epochs):
            average_train_loss = 0
            average_val_loss = 0
            average_train_accuracy = 0
            average_val_accuracy = 0
            average_val_f1_score = 0
            print('EPOCH {0}/{1}...\n'.format(ep + 1, epochs))
            with tqdm(total=num_interactions) as t:
                for batch in shuffle_extract_batch(train_edge_index_df, train_neg_edges, batch_size):
                    current_loss_train, train_accuracy = self.train_step(batch)
                    val_dict = self.predict(val_edge_index_df.to_numpy(), np.array(val_neg_edges))
                    average_train_loss += current_loss_train
                    average_train_accuracy += train_accuracy
                    average_val_loss += val_dict['loss']
                    average_val_accuracy += val_dict['accuracy']
                    average_val_f1_score += val_dict['f1_score']
                    t.set_postfix({'train_loss': f'{current_loss_train:.5f}',
                                   'train_accuracy': f'{train_accuracy:.5f}',
                                   'val_loss': f'{val_dict["loss"]:.5f}',
                                   'val_accuracy': f'{val_dict["accuracy"]:.5f}',
                                   'val_f1_score': f'{val_dict["f1_score"]: .5f}'})
                    t.update()
            all_train_losses += [average_train_loss / num_interactions]
            all_val_losses += [average_val_loss / num_interactions]
            val_accuracies.append(average_val_accuracy / num_interactions)
            print(f'\nEND EPOCH {ep + 1}.'
                  f'\tAvg train loss: {average_train_loss / num_interactions:.3f}'
                  f'\tAvg train accuracy: {average_train_accuracy / num_interactions:.3f}'
                  f'\tAvg val loss: {average_val_loss / num_interactions:.3f}'
                  f'\tAvg val accuracy: {average_val_accuracy / num_interactions:.3f}'
                  f'\tAvg val f1 score: {average_val_f1_score / num_interactions:.3f}\n')
            all_train_accuracies += [average_train_accuracy / num_interactions]
            #all_train_accuracies += [average_val_accuracy / num_interactions]
            all_val_accuracies += [average_val_accuracy / num_interactions]

            if best_val_accuracy < average_val_accuracy / num_interactions:
                best_epoch = ep + 1
                best_val_accuracy = average_val_accuracy / num_interactions
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
        if validation:
            emb_set_node_one_update, emb_set_node_two_update = self.propagate_embeddings(extend_adj=False)
        else:
            emb_set_node_one_update, emb_set_node_two_update = self.propagate_embeddings(extend_adj=True)
        current_set_node_one_pos, pos_set_node_two = inputs_pos[:, 0], inputs_pos[:, 1]
        current_set_node_one_neg, neg_set_node_two = inputs_neg[:, 0], inputs_neg[:, 1]
        current_set_node_one_emb_pos = emb_set_node_one_update[current_set_node_one_pos]
        current_set_node_one_emb_neg = emb_set_node_one_update[current_set_node_one_neg]
        pos_set_node_two_emb = emb_set_node_two_update[pos_set_node_two]
        neg_set_node_two_emb = emb_set_node_two_update[neg_set_node_two]

        out_pos = torch.sigmoid(self.dense_network(current_set_node_one_emb_pos * pos_set_node_two_emb))
        out_neg = torch.sigmoid(self.dense_network(current_set_node_one_emb_neg * neg_set_node_two_emb))

        labels = torch.cat([torch.ones(out_pos.shape[0]), torch.zeros(out_neg.shape[0])])
        batch_loss = torch.nn.functional.binary_cross_entropy(
            torch.cat([out_pos, out_neg])[:, 0].to(device),
            labels.to(device))

        batch_loss += self.reg_term * (1 / 2) * (current_set_node_one_emb_pos.norm(2).pow(2) +
                                                 current_set_node_one_emb_neg.norm(2).pow(2) +
                                                 pos_set_node_two_emb.norm(2).pow(2) +
                                                 neg_set_node_two_emb.norm(2).pow(2)) / out_pos.shape[0]

        preds = torch.cat([out_pos, out_neg], dim=0).reshape(-1).detach().cpu().numpy()
        preds_labels = (preds >= 0.5).astype(int)
        labels = labels.detach().cpu().numpy().astype(int)

        if validation:
            return {
                'loss': batch_loss.detach().cpu().numpy().item(),
                'accuracy': accuracy_score(y_true=labels, y_pred=preds_labels),
                'f1_score': f1_score(y_true=labels, y_pred=preds_labels, zero_division=0)
            }
        else:
            results = classification_report(y_true=labels, y_pred=preds_labels, output_dict=True, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds_labels).ravel()
            return {
                'precision_0': results['0']['precision'],
                'recall_0': results['0']['recall'],
                'f1_score_0': results['0']['f1-score'],
                'precision_1': results['1']['precision'],
                'recall_1': results['1']['recall'],
                'f1_score_1': results['1']['f1-score'],
                'accuracy': results['accuracy'],
                'auc': roc_auc_score(y_true=labels, y_score=preds),
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            }

    def get_params(self):
        return {
            "emb_size": self.emb_size,
            "learning_rate": self.learning_rate,
            "reg_term": self.reg_term,
            "freeze_node_one": self.freeze_node_one,
            "freeze_node_two": self.freeze_node_two,
        }
