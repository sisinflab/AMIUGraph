random_seed = 42
epochs = 1000
patience = 50

model_parameters_kge = dict()
model_parameters_gnn = dict()


# GCNConv
# model_parameters_gnn['GNNAccHetero'] = {
#     "emb_size": [64],
#     "dense_weight_size": [32],
#     "learning_rate": [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
#     "reg_term": [1e-7, 1e-5, 1e-3],
#     "aggregation": ['sum', 'mean', 'max', 'median', 'min'],
#     "num_layers": [2, 3],
#     "gnn": ['GCNConv'],
#     "gnn_parameters": {
#         "in_channels": [64],
#         "out_channels": [64],
#         "add_self_loops": [False],
#         "normalize": [True]
#     }
# }

# SAGEConv
# model_parameters_gnn['GNNAccHetero'] = {
#     "emb_size": [64],
#     "dense_weight_size": [32],
#     "learning_rate": [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
#     "reg_term": [1e-7, 1e-5, 1e-3],
#     "aggregation": ['sum', 'mean', 'max', 'median', 'min'],
#     "num_layers": [1],
#     "gnn": ['SAGEConv'],
#     "gnn_parameters": {
#         "in_channels": [64],
#         "out_channels": [64],
#         "root_weight": [True],
#         "normalize": [True],
#         "project": [True],
#         "add_self_loops": [False],
#         "bias": [True]
#     }
# }


# TODO
# Adam
model_parameters_kge['ConvE'] = {
  "emb_size": [64, 128, 256, 512],
  "learning_rate": [0.001, 0.003],
  "reg_term": [1e-7, 1e-5, 1e-3],
  "first_shape_2d": [16],
}

## AdaGrad
#model_parameters_kge['ComplEx'] = {
#  "emb_size": [64, 128, 256, 512],
#  "learning_rate": [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
#  "reg_term": [1e-7, 1e-5, 1e-3]
#}
#
#model_parameters_kge['DistMult'] = {
#  "emb_size": [64, 128, 256, 512],
#  "learning_rate": [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
#  "reg_term": [1e-7, 1e-5, 1e-3]
#}
#
#model_parameters_kge['CP'] = {
#  "emb_size": [64, 128, 256, 512],
#  "learning_rate": [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
#  "reg_term": [1e-7, 1e-5, 1e-3]
#}
#
#model_parameters_gnn['GCN'] = {
#    "emb_size": [64],
#    "weight_size": [[64, 64], [64, 64, 64]],
#    "dense_weight_size": [[32]],
#    "learning_rate": [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
#    "reg_term": [1e-7, 1e-5, 1e-3],
#    "normalize": [True],
#    "freeze_node_one": [False]
# }
#
#model_parameters_gnn['GIN'] = {
#    "emb_size": [64],
#    "nn_weight_size": [[64], [64, 64]],
#    "num_layers": [2, 3],
#    "dense_weight_size": [[32]],
#    "learning_rate": [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
#    "reg_term": [1e-7, 1e-5, 1e-3],
#    "eps": [0.0],
#    "train_eps": [True],
#    "freeze_node_one": [False]
#}
#
#model_parameters_gnn['GAT'] = {
#    "emb_size": [64],
#    "weight_size": [[64], [64, 64]],
#    "dense_weight_size": [[32]],
#    "learning_rate": [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
#    "reg_term": [1e-7, 1e-5, 1e-3],
#    "heads": [4, 8],
#    "freeze_node_one": [False]
#}
#
#model_parameters_gnn['SAGE'] = {
#    "emb_size": [64],
#    "weight_size": [[64], [64, 64]],
#    "dense_weight_size": [[32]],
#    "learning_rate": [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
#    "reg_term": [1e-7, 1e-5, 1e-3],
#    "normalize": [True],
#    "root_weight": [True],
#    "project": [False],
#    "freeze_node_one": [False]
#}

#model_parameters['NNLinker-Spectral'] = {
#    "embedding_method": ['Spectral'],
#    "n_neighbors": [3, 4, 5],
#    "threshold": [0.5, 0.3, 0.7],
#}
#
#model_parameters['NNLinker-SVD'] = {
#    "embedding_method": ['SVD'],
#    "n_neighbors": [3, 4, 5],
#    "threshold": [0.5, 0.3, 0.7],
#}
