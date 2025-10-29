import torch
from collections import OrderedDict
import torch_geometric
import numpy as np
from torch_geometric.nn import GINConv
from sklearn.metrics import accuracy_score

from models.base import GNNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GIN(GNNModel):
    def __init__(self, conf, *args, **kwargs):
        super(GIN, self).__init__(conf=conf, *args, **kwargs)

        self.nn_weight_size_list = [self.emb_size] + conf["nn_weight_size"]
        self.dense_weight_size_list = [self.nn_weight_size_list[-1]] + conf["dense_weight_size"] + [1]
        self.nn_num_layers = len(self.nn_weight_size_list) - 1
        self.num_layers = conf["num_layers"]
        self.num_layers_out = len(self.dense_weight_size_list) - 1
        self.eps = conf["eps"]
        self.train_eps = conf["train_eps"]

        propagation_network_list = []

        for layer in range(self.num_layers):
            nn_list = []
            for layer_nn in range(self.nn_num_layers):
                nn_list.append(
                    ('nn_' + str(layer) + '_' + str(layer_nn),
                     torch.nn.Linear(in_features=self.nn_weight_size_list[layer_nn],
                                     out_features=self.nn_weight_size_list[layer_nn + 1],
                                     bias=True)))
            current_nn = torch.nn.Sequential(OrderedDict(nn_list))
            current_nn.to(device)

            propagation_network_list.append((GINConv(
                nn=current_nn,
                eps=self.eps,
                train_eps=self.train_eps
            ), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(device)

        dense_network_list = []
        for layer in range(self.num_layers_out):
            dense_network_list.append(
                ('dense_' + str(layer), torch.nn.Linear(in_features=self.dense_weight_size_list[layer],
                                                        out_features=self.dense_weight_size_list[layer + 1],
                                                        bias=True)))
        self.dense_network = torch.nn.Sequential(OrderedDict(dense_network_list))
        self.dense_network.to(device)

    def get_trainable_parameters(self):
        trainable_params = list(self.propagation_network.parameters()) + \
               list(self.dense_network.parameters()) + \
               [self.emb_set_node_one, self.emb_set_node_two]

        if self.projection_node_one:
            trainable_params.extend(list(self.projection_node_one.parameters()))
        if self.projection_node_two:
            trainable_params.extend(list(self.projection_node_two.parameters()))

        return trainable_params

    def _propagate_embeddings(self, all_node_embeddings, adj):

        for current_layer in range(self.num_layers):
            all_node_embeddings = torch.relu(
                list(self.propagation_network.children())[current_layer](all_node_embeddings.to(device),
                                                                         adj))
            all_node_embeddings = torch.nn.functional.normalize(all_node_embeddings, p=2, dim=-1)

        emb_node_set_one, emb_node_set_two = torch.split(all_node_embeddings, [self.num_set_node_one,
                                                                               self.num_set_node_two], 0)
        return emb_node_set_one, emb_node_set_two

    def train_step(self, inputs):
        emb_set_node_one_update, emb_set_node_two_update = self.propagate_embeddings()
        node_one = np.array(inputs[0].tolist())
        node_two = np.array(inputs[1].tolist())

        out = torch.sigmoid(self.dense_network(
            emb_set_node_one_update[node_one] * emb_set_node_two_update[node_two])
        )

        labels = torch.tensor(inputs[2].tolist(), dtype=torch.float32, device=device)
        batch_loss = torch.nn.functional.binary_cross_entropy(
            out[:, 0].to(device),
            labels.to(device))

        batch_loss += self.reg_term * (1 / 2) * (emb_set_node_one_update[node_one].norm(2).pow(2) +
                                                 emb_set_node_two_update[node_two].norm(2).pow(2)) / out.shape[0]

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        preds = out.reshape(-1).detach().cpu().numpy()
        preds_labels = (preds >= 0.5).astype(int)
        labels = labels.detach().cpu().numpy().astype(int)

        return batch_loss.detach().cpu().numpy().item(), accuracy_score(y_true=labels, y_pred=preds_labels)

    def get_params(self):
        super().get_params() + \
        {
            "nn_weight_size_list": self.nn_weight_size_list,
            "dense_weight_size_list": self.dense_weight_size_list,
            "num_layers": self.num_layers,
            "nn_num_layers": self.nn_num_layers,
            "num_layers_out": self.num_layers_out,
            "eps": self.eps,
            "train_eps": self.train_eps
        }
