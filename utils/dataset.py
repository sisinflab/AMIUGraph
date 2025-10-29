import pandas as pd
from operator import itemgetter
from torch_sparse import SparseTensor
from scipy import sparse as sp
import torch
import numpy as np
import os
import random
from torch_geometric.data import HeteroData
from typing import List
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def shuffle_extract_batch(dataset_pos, dataset_neg, bs):
    pos_df = dataset_pos.copy()
    neg_df = pd.DataFrame(dataset_neg).copy()
    pos_df[2] = pd.Series([1] * len(pos_df))
    neg_df[2] = pd.Series([0] * len(neg_df))
    shuffled_dataset = pd.concat([pos_df.copy(), neg_df.copy()], axis=0).sample(frac=1).reset_index(drop=True)
    len_dataset = len(shuffled_dataset) // 2
    shuffled_dataset = shuffled_dataset[:len_dataset]

    for index, offset in enumerate(range(0, len_dataset, bs)):
        offset_stop = min(offset + bs, len_dataset)
        current_batch = shuffled_dataset.iloc[offset:offset_stop].copy().reset_index(drop=True)
        yield current_batch


def shuffle_extract_batch_kg(dataset_pos, dataset_neg, bs):
    pos_df = dataset_pos.copy()
    neg_df = pd.DataFrame(dataset_neg).copy()
    pos_df[3] = pd.Series([1] * len(pos_df))
    neg_df[3] = pd.Series([0] * len(neg_df))
    shuffled_dataset = pd.concat([pos_df.copy(), neg_df.copy()], axis=0).sample(frac=1).reset_index(drop=True)
    len_dataset = len(shuffled_dataset) // 2
    shuffled_dataset = shuffled_dataset[:len_dataset]

    for index, offset in enumerate(range(0, len_dataset, bs)):
        offset_stop = min(offset + bs, len_dataset)
        current_batch = shuffled_dataset.iloc[offset:offset_stop].copy().reset_index(drop=True)
        yield current_batch


# def generate_graph_hetero(all_data):
#     data = HeteroData()
#     idx = 1
#     num_nodes = dict()
#     relations = list()
#
#     for key, df in all_data.items():
#         df.columns = [df.columns[0].replace('_ID', ''), df.columns[1].replace('_ID', '')]
#
#         if not df.columns[0] in num_nodes:
#             num_nodes[df.columns[0]] = df[df.columns[0]].nunique()
#         if not df.columns[1] in num_nodes:
#             num_nodes[df.columns[1]] = df[df.columns[1]].nunique()
#
#         map_node_one = {private: public for public, private in enumerate(df[df.columns[0]].unique())}
#         map_node_two = {private: public + num_nodes[df.columns[0]] for public, private in
#                         enumerate(df[df.columns[1]].unique())}
#
#         rows = [map_node_one[n] for n in df[df.columns[0]].tolist()]
#         cols = [map_node_two[n] for n in df[df.columns[1]].tolist()]
#
#         edge_index = torch.tensor([rows, cols], dtype=torch.long)
#
#         data[df.columns[0], f'rel{idx}', df.columns[1]].edge_index = SparseTensor(
#             row=edge_index[0],
#             col=edge_index[1],
#             sparse_sizes=(df[df.columns[0]].nunique() + df[df.columns[1]].nunique(),
#                           df[df.columns[0]].nunique() + df[df.columns[1]].nunique())
#         )
#         data[df.columns[1], f'rel{idx + 1}', df.columns[0]].edge_index = SparseTensor(
#             row=edge_index[1],
#             col=edge_index[0],
#             sparse_sizes=(df[df.columns[0]].nunique() + df[df.columns[1]].nunique(),
#                           df[df.columns[0]].nunique() + df[df.columns[1]].nunique())
#         )
#
#         relations.append((df.columns[0], f'rel{idx}', df.columns[1]))
#         relations.append((df.columns[1], f'rel{idx + 1}', df.columns[0]))
#
#         idx += 2
#
#     return data, num_nodes, relations


def generate_train_hetero(data_path, random_seed):
    random.seed(random_seed)

    df = pd.read_csv(data_path, sep='\t')

    train_edge_index_df = df.sample(frac=0.8, random_state=random_seed)
    val_edge_index_df = train_edge_index_df.sample(frac=0.1, random_state=random_seed)
    train_edge_index_df = train_edge_index_df[~train_edge_index_df.index.isin(val_edge_index_df.index)]

    return train_edge_index_df, val_edge_index_df


def generate_datasets_kg(data_path: List, random_seed, as_hetero=False):
    random.seed(random_seed)

    # df = pd.read_csv(data_path, sep='\t')
    # df.columns = ['column_one', 'column_two', 'relations']
    train_edge_index_df = {}
    val_edge_index_df = {}
    test_edge_index_df = {}
    train_neg_edge_index_df = {}
    val_neg_edge_index_df = {}
    test_neg_edge_index_df = {}
    for path in data_path:
        train_edge_index_df[path] = pd.read_csv(f"data/{path}/train.tsv", sep='\t', header=None)
        train_edge_index_df[path][2] = pd.Series([path] * len(train_edge_index_df[path]))
        train_edge_index_df[path].columns = ["head", "tail", "rel"]
        val_edge_index_df[path] = pd.read_csv(f"data/{path}/val.tsv", sep='\t', header=None)
        val_edge_index_df[path][2] = pd.Series([path] * len(val_edge_index_df[path]))
        val_edge_index_df[path].columns = ["head", "tail", "rel"]
        test_edge_index_df[path] = pd.read_csv(f"data/{path}/test.tsv", sep='\t', header=None)
        test_edge_index_df[path][2] = pd.Series([path] * len(test_edge_index_df[path]))
        test_edge_index_df[path].columns = ["head", "tail", "rel"]

        train_neg_edge_index_df[path] = pd.read_csv(f"data/{path}/train_neg.tsv", sep='\t', header=None)
        train_neg_edge_index_df[path][2] = pd.Series([path] * len(train_neg_edge_index_df[path]))
        train_neg_edge_index_df[path].columns = ["head", "tail", "rel"]
        val_neg_edge_index_df[path] = pd.read_csv(f"data/{path}/val_neg.tsv", sep='\t', header=None)
        val_neg_edge_index_df[path][2] = pd.Series([path] * len(val_neg_edge_index_df[path]))
        val_neg_edge_index_df[path].columns = ["head", "tail", "rel"]
        test_neg_edge_index_df[path] = pd.read_csv(f"data/{path}/test_neg.tsv", sep='\t', header=None)
        test_neg_edge_index_df[path][2] = pd.Series([path] * len(test_neg_edge_index_df[path]))
        test_neg_edge_index_df[path].columns = ["head", "tail", "rel"]

    df_train = pd.concat(list(train_edge_index_df.values()), axis=0)
    df_val = pd.concat(list(val_edge_index_df.values()), axis=0)
    df_test = pd.concat(list(test_edge_index_df.values()), axis=0)
    df_train_neg_hard = pd.concat(list(train_neg_edge_index_df.values()), axis=0)
    df_val_neg = pd.concat(list(val_neg_edge_index_df.values()), axis=0)
    df_test_neg = pd.concat(list(test_neg_edge_index_df.values()), axis=0)

    unique_nodes = set(list(df_train["head"].unique()) + list(df_train["tail"].unique()))
    num_relations = len(data_path)
    relations_train = df_train["rel"].unique().tolist()

    node_public_private = {n: idx for idx, n in enumerate(unique_nodes)}
    relations_public_private = {r: idx for idx, r in enumerate(relations_train)}

    train_rows = df_train["head"].tolist()
    train_cols = df_train["tail"].tolist()
    train_rel = df_train["rel"].tolist()

    val_rows = df_val["head"].tolist()
    val_cols = df_val["tail"].tolist()
    val_rel = df_val["rel"].tolist()

    test_rows = df_test["head"].tolist()
    test_cols = df_test["tail"].tolist()
    test_rel = df_test["rel"].tolist()

    train_neg_rows = df_train_neg_hard["head"].tolist()
    train_neg_cols = df_train_neg_hard["tail"].tolist()
    train_neg_rel = df_train_neg_hard["rel"].tolist()

    val_neg_rows = df_val_neg["head"].tolist()
    val_neg_cols = df_val_neg["tail"].tolist()
    val_neg_rel = df_val_neg["rel"].tolist()

    test_neg_rows = df_test_neg["head"].tolist()
    test_neg_cols = df_test_neg["tail"].tolist()
    test_neg_rel = df_test_neg["rel"].tolist()

    df_train = pd.concat([pd.Series(itemgetter(*train_rows)(node_public_private)),
                          pd.Series(itemgetter(*train_cols)(node_public_private)),
                          pd.Series(itemgetter(*train_rel)(relations_public_private))], axis=1)

    df_val = pd.concat([pd.Series(itemgetter(*val_rows)(node_public_private)),
                        pd.Series(itemgetter(*val_cols)(node_public_private)),
                        pd.Series(itemgetter(*val_rel)(relations_public_private))], axis=1)

    df_test = pd.concat([pd.Series(itemgetter(*test_rows)(node_public_private)),
                         pd.Series(itemgetter(*test_cols)(node_public_private)),
                         pd.Series(itemgetter(*test_rel)(relations_public_private))], axis=1)

    df_train_neg_hard = pd.concat([pd.Series(itemgetter(*train_neg_rows)(node_public_private)),
                                   pd.Series(itemgetter(*train_neg_cols)(node_public_private)),
                                   pd.Series(itemgetter(*train_neg_rel)(relations_public_private))], axis=1)

    df_val_neg = pd.concat([pd.Series(itemgetter(*val_neg_rows)(node_public_private)),
                            pd.Series(itemgetter(*val_neg_cols)(node_public_private)),
                            pd.Series(itemgetter(*val_neg_rel)(relations_public_private))], axis=1)

    df_test_neg = pd.concat([pd.Series(itemgetter(*test_neg_rows)(node_public_private)),
                             pd.Series(itemgetter(*test_neg_cols)(node_public_private)),
                             pd.Series(itemgetter(*test_neg_rel)(relations_public_private))], axis=1)

    train_neg = []
    for idx, row in df_train.iterrows():
        s_o = random.sample([0, 1], 1)[0]
        if s_o == 0:
            pos = df_train[(df_train[1] == row[1]) & (df_train[2] == row[2])][
                0].tolist()
            neg = random.sample(set(node_public_private.values()).difference(pos), 1)[0]
            train_neg.append((neg, row[1], row[2]))
        else:
            pos = df_train[(df_train[0] == row[0]) & (df_train[2] == row[2])][
                1].tolist()
            neg = random.sample(set(node_public_private.values()).difference(pos), 1)[0]
            train_neg.append((row[0], neg, row[2]))
    df_train_neg_soft = pd.DataFrame(train_neg)
    # df_train_neg = pd.concat([df_train_neg_soft, df_train_neg_hard], axis=0)
    df_train_neg = df_train_neg_hard
    df_train_neg = df_train_neg.drop_duplicates()
    df_train_neg = df_train_neg.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    if as_hetero:
        hetero_data = HeteroData()
        nodes = dict()
        for rel_name, rel_id in relations_public_private.items():
            head = rel_name.split('_')[0]
            tail = rel_name.split('_')[1]
            nodes.setdefault(head, [])
            nodes.setdefault(tail, [])
            nodes[head] += list(df_train[df_train[2] == rel_id][0].unique())
            nodes[tail] += list(df_train[df_train[2] == rel_id][1].unique())
        num_nodes = {key: len(set(l)) for key, l in nodes.items()}
        nodes_mapping = {node_type: {old_value: new_value for new_value, old_value in enumerate(set(current_node))} for
                         node_type, current_node in nodes.items()}
        for rel_name, rel_id in relations_public_private.items():
            head = rel_name.split('_')[0]
            tail = rel_name.split('_')[1]
            rel = rel_name
            edge_index_df = df_train[df_train[2] == rel_id][[0, 1]]
            edge_index_df[0] = edge_index_df[0].map(nodes_mapping[head])
            edge_index_df[1] = edge_index_df[1].map(nodes_mapping[tail])

            df_train.loc[df_train[df_train[2] == rel_id].index, 0] = df_train[df_train[2] == rel_id][0].map(
                nodes_mapping[head])
            df_train.loc[df_train[df_train[2] == rel_id].index, 1] = df_train[df_train[2] == rel_id][1].map(
                nodes_mapping[tail])

            df_val.loc[df_val[df_val[2] == rel_id].index, 0] = df_val[df_val[2] == rel_id][0].map(nodes_mapping[head])
            df_val.loc[df_val[df_val[2] == rel_id].index, 1] = df_val[df_val[2] == rel_id][1].map(nodes_mapping[tail])
            df_test.loc[df_test[df_test[2] == rel_id].index, 0] = df_test[df_test[2] == rel_id][0].map(
                nodes_mapping[head])
            df_test.loc[df_test[df_test[2] == rel_id].index, 1] = df_test[df_test[2] == rel_id][1].map(
                nodes_mapping[tail])
            df_train_neg.loc[df_train_neg[df_train_neg[2] == rel_id].index, 0] = \
                df_train_neg[df_train_neg[2] == rel_id][0].map(nodes_mapping[head])
            df_train_neg.loc[df_train_neg[df_train_neg[2] == rel_id].index, 1] = \
                df_train_neg[df_train_neg[2] == rel_id][1].map(nodes_mapping[tail])
            df_val_neg.loc[df_val_neg[df_val_neg[2] == rel_id].index, 0] = df_val_neg[df_val_neg[2] == rel_id][0].map(
                nodes_mapping[head])
            df_val_neg.loc[df_val_neg[df_val_neg[2] == rel_id].index, 1] = df_val_neg[df_val_neg[2] == rel_id][1].map(
                nodes_mapping[tail])
            df_test_neg.loc[df_test_neg[df_test_neg[2] == rel_id].index, 0] = df_test_neg[df_test_neg[2] == rel_id][
                0].map(nodes_mapping[head])
            df_test_neg.loc[df_test_neg[df_test_neg[2] == rel_id].index, 1] = df_test_neg[df_test_neg[2] == rel_id][
                1].map(nodes_mapping[tail])

            edge_index = torch.tensor(edge_index_df.to_numpy().transpose(), dtype=torch.long)
            hetero_data[head, rel, tail].edge_index = edge_index

        return df_train, \
               df_val, \
               df_test, \
               df_train_neg, \
               df_val_neg, \
               df_test_neg, \
               len(unique_nodes), \
               num_relations, \
               relations_train, \
               hetero_data, \
               num_nodes, \
               nodes

    return df_train, \
           df_val, \
           df_test, \
           df_train_neg, \
           df_val_neg, \
           df_test_neg, \
           len(unique_nodes), \
           num_relations, \
           relations_train


def generate_bipartite_datasets_for_inductive(data_path,
                                              random_seed,
                                              save_df=False,
                                              target_col=0,
                                              unseen_ratio=0.2,
                                              ind_test_ratio=0.1,
                                              return_scaler=False,
                                              ):
    # data = generate_datasets(data_path, random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    path = f'data/{data_path}/dataset.tsv'
    df = pd.read_csv(path, sep='\t')
    columns = df.columns
    for col in columns:
        df[col] = df[col].map(lambda x: col.split('_')[0] + '_' + str(x))
    df.columns = ['column_one', 'column_two']
    targets = df[df.columns[target_col]].unique()

    ind_entities = np.random.choice(targets,
                                    size=int(unseen_ratio * len(targets)))

    # split G_ind and G_ind(test) based on interactions

    df_trans = df[df[df.columns[target_col]].isin(np.setdiff1d(targets, ind_entities))]
    df_ind = df[df[df.columns[target_col]].isin(ind_entities)]

    if unseen_ratio > 0:
        g_trans = split_df(df_trans, path, random_seed, save_df=save_df, val_ratio=0.0)
    else:
        g_trans = split_df(df_trans, path, random_seed, save_df=save_df)

    for i in range(6):
        if len(g_trans[i]) > 0:
            g_trans[i][0] = g_trans[i][0].map(g_trans[-1]["node_one_private_public"])
            g_trans[i][1] = g_trans[i][1].map(g_trans[-1]["node_two_private_public"])

    nodes_set_one = g_trans[0][0].unique().tolist()
    nodes_set_two = g_trans[0][1].unique().tolist()
    g_ind = None
    if unseen_ratio > 0:
        g_ind = split_df(df_ind, f"{path}_ind", random_seed, train_ratio=1 - ind_test_ratio, val_ratio=0.0,
                         save_df=save_df)
        for i in range(6):
            if len(g_ind[i]) > 0:
                g_ind[i][0] = g_ind[i][0].map(g_ind[-1]["node_one_private_public"])
                g_ind[i][1] = g_ind[i][1].map(g_ind[-1]["node_two_private_public"])

        nodes_set_one += (g_ind[0][0].unique().tolist())
        nodes_set_two += (g_ind[0][1].unique().tolist())

    nodes_set_one = np.unique(nodes_set_one)
    nodes_set_two = np.unique(nodes_set_two)
    num_nodes_set_one = len(nodes_set_one)
    num_nodes_set_two = len(nodes_set_two)

    node_one_public_private = {p: idx for idx, p in enumerate(nodes_set_one)}
    node_two_public_private = {d: idx for idx, d in enumerate(nodes_set_two)}
    node_one_private_public = {idx: p for idx, p in enumerate(nodes_set_one)}
    #node_two_private_public = {idx: d for idx, d in enumerate(nodes_set_two)}



    patient_attributes = get_patient_attributes()[0]
    patient_attributes_mapping = get_patient_attributes()[2]
    patient_attributes_scaler = get_patient_attributes()[3]
    attributes = [patient_attributes[patient_attributes_mapping[node_one_private_public[i]]] for i in range(num_nodes_set_one)]

    #curr_patient_attributes = [patient_attributes[p_id, :] for p_id in nodes_set_one_public_ids]
    #curr_patient_attributes = patient_attributes[[nodes_set_one_public_ids], :]

    for i in range(6):
        if len(g_trans[i]) > 0:
            g_trans[i] = pd.concat([pd.Series(itemgetter(*g_trans[i][0].tolist())(node_one_public_private)),
                                    pd.Series(itemgetter(*g_trans[i][1].tolist())(node_two_public_private))], axis=1)

    g_trans_train_rows = g_trans[0][0].tolist()
    g_trans_train_cols = g_trans[0][1].tolist()
    g_trans_train_cols = [col + num_nodes_set_one for col in g_trans_train_cols]

    g_trans_edge_index = torch.tensor(
        [g_trans_train_rows + g_trans_train_cols,
         g_trans_train_cols + g_trans_train_rows], dtype=torch.long)

    adj_train = SparseTensor(row=g_trans_edge_index[0],
                             col=g_trans_edge_index[1],
                             sparse_sizes=(
                                 num_nodes_set_one + num_nodes_set_two,
                                 num_nodes_set_one + num_nodes_set_two))
    adj = adj_train




    if unseen_ratio > 0:
        for i in range(6):
            if len(g_ind[i]) > 0:
                g_ind[i] = pd.concat([
                    pd.Series(itemgetter(*g_ind[i][0].tolist())(node_one_public_private)),
                    pd.Series(itemgetter(*g_ind[i][1].tolist())(node_two_public_private))], axis=1)

        g_ind_train_rows = g_ind[0][0].tolist()
        g_ind_train_cols = g_ind[0][1].tolist()

        g_ind_edge_index = torch.tensor(
            [g_trans_train_rows + g_trans_train_cols + g_ind_train_rows + g_ind_train_cols,
             g_trans_train_cols + g_trans_train_rows + g_ind_train_cols + g_ind_train_rows], dtype=torch.long)

        adj_expanded = SparseTensor(row=g_ind_edge_index[0],
                                    col=g_ind_edge_index[1],
                                    sparse_sizes=(
                                        num_nodes_set_one + num_nodes_set_two,
                                        num_nodes_set_one + num_nodes_set_two))

        adj = (adj_train, adj_expanded)

    if return_scaler:
        return g_trans, g_ind, num_nodes_set_one, num_nodes_set_two, adj, attributes, nodes_set_one, nodes_set_two, patient_attributes_scaler
    else:
        return g_trans, g_ind, num_nodes_set_one, num_nodes_set_two, adj, attributes, nodes_set_one, nodes_set_two


def split_df(df, path, random_seed, train_ratio=0.8, val_ratio=0.1, save_df=False):
    train_edge_index_df = df.sample(frac=train_ratio, random_state=random_seed)
    val_edge_index_df = train_edge_index_df.sample(frac=val_ratio, random_state=random_seed)
    test_edge_index_df = df[~df.index.isin(train_edge_index_df.index)].reset_index(drop=True)

    train_edge_index_df = train_edge_index_df[~train_edge_index_df.index.isin(val_edge_index_df.index)].reset_index(
        drop=True)
    train_edge_index_df.reset_index(drop=True, inplace=True)
    val_edge_index_df.reset_index(drop=True, inplace=True)

    node_one_train = train_edge_index_df['column_one'].unique().tolist()
    node_two_train = train_edge_index_df['column_two'].unique().tolist()

    val_edge_index_df = val_edge_index_df[val_edge_index_df['column_one'].isin(node_one_train)]
    val_edge_index_df = val_edge_index_df[val_edge_index_df['column_two'].isin(node_two_train)]
    test_edge_index_df = test_edge_index_df[test_edge_index_df['column_one'].isin(node_one_train)]
    test_edge_index_df = test_edge_index_df[test_edge_index_df['column_two'].isin(node_two_train)]

    if save_df:
        train_edge_index_df.to_csv(os.path.dirname(path) + '/train.tsv', sep='\t', header=None, index=None)
        val_edge_index_df.to_csv(os.path.dirname(path) + '/val.tsv', sep='\t', header=None, index=None)
        test_edge_index_df.to_csv(os.path.dirname(path) + '/test.tsv', sep='\t', header=None, index=None)

    node_one_public_private = {p: idx for idx, p in enumerate(node_one_train)}
    node_two_public_private = {d: idx for idx, d in enumerate(node_two_train)}
    node_one_private_public = {idx: p for idx, p in enumerate(node_one_train)}
    node_two_private_public = {idx: d for idx, d in enumerate(node_two_train)}

    train_rows = train_edge_index_df['column_one'].tolist()
    train_cols = train_edge_index_df['column_two'].tolist()

    val_rows = val_edge_index_df['column_one'].tolist()
    val_cols = val_edge_index_df['column_two'].tolist()

    test_rows = test_edge_index_df['column_one'].tolist()
    test_cols = test_edge_index_df['column_two'].tolist()

    train_edge_index_df = pd.concat([pd.Series(itemgetter(*train_rows)(node_one_public_private)),
                                     pd.Series(itemgetter(*train_cols)(node_two_public_private))], axis=1)

    if val_ratio > 0:
        val_edge_index_df = pd.concat([pd.Series(itemgetter(*val_rows)(node_one_public_private)),
                                       pd.Series(itemgetter(*val_cols)(node_two_public_private))], axis=1)
    else:
        val_edge_index_df = []

    test_edge_index_df = pd.concat([pd.Series(itemgetter(*test_rows)(node_one_public_private)),
                                    pd.Series(itemgetter(*test_cols)(node_two_public_private))], axis=1)

    neg_rows, neg_cols = np.where(1 - sp.coo_matrix((np.ones(len(train_rows)), (train_edge_index_df[0].tolist(),
                                                                                train_edge_index_df[
                                                                                    1].tolist()))).todense() != 0)
    neg_edges = list(zip(neg_rows.tolist(), neg_cols.tolist()))
    random.shuffle(neg_edges)

    train_neg_edges = pd.DataFrame(neg_edges[:len(train_edge_index_df)])
    val_neg_edges = pd.DataFrame(neg_edges[len(train_edge_index_df):
                                           len(train_edge_index_df) +
                                           len(val_edge_index_df)])

    test_neg_edges = pd.DataFrame(neg_edges[len(train_edge_index_df) + len(val_edge_index_df):
                                            len(train_edge_index_df) +
                                            len(val_edge_index_df) +
                                            len(test_edge_index_df)])

    train_neg_edges_public = train_neg_edges.copy()
    val_neg_edges_public = val_neg_edges.copy()
    test_neg_edges_public = test_neg_edges.copy()

    train_neg_edges_public[0] = train_neg_edges_public[0].map(node_one_private_public)
    train_neg_edges_public[1] = train_neg_edges_public[1].map(node_two_private_public)

    if val_ratio > 0:
        val_neg_edges_public[0] = val_neg_edges_public[0].map(node_one_private_public)
        val_neg_edges_public[1] = val_neg_edges_public[1].map(node_two_private_public)

    test_neg_edges_public[0] = test_neg_edges_public[0].map(node_one_private_public)
    test_neg_edges_public[1] = test_neg_edges_public[1].map(node_two_private_public)

    if save_df:
        train_neg_edges_public.to_csv(os.path.dirname(path) + '/train_neg.tsv', sep='\t', header=None, index=None)
        if val_ratio > 0:
            val_neg_edges_public.to_csv(os.path.dirname(path) + '/val_neg.tsv', sep='\t', header=None, index=None)
        test_neg_edges_public.to_csv(os.path.dirname(path) + '/test_neg.tsv', sep='\t', header=None, index=None)

    mappings = {"node_one_private_public": node_one_private_public,
                "node_two_private_public": node_two_private_public,
                "node_one_public_private": node_one_public_private,
                "node_two_public_private": node_two_public_private}

    if val_ratio == 0:
        val_edge_index_df = []
        val_neg_edges = []

    return [
        train_edge_index_df,
        val_edge_index_df,
        test_edge_index_df,
        train_neg_edges,
        val_neg_edges,
        test_neg_edges,
        mappings
    ]



def generate_datasets(data_path, random_seed, save_df=False):
    """
    Generates bipartite graphs for GNNs
    """

    random.seed(random_seed)
    path = f'data/{data_path}/dataset.tsv'
    df = pd.read_csv(path, sep='\t')
    columns = df.columns

    for col in columns:
        df[col] = df[col].map(lambda x: col.split('_')[0] + '_' + str(x))
    df.columns = ['column_one', 'column_two']
    train_edge_index_df = df.sample(frac=0.8, random_state=random_seed)
    val_edge_index_df = train_edge_index_df.sample(frac=0.1, random_state=random_seed)
    test_edge_index_df = df[~df.index.isin(train_edge_index_df.index)].reset_index(drop=True)
    train_edge_index_df = train_edge_index_df[~train_edge_index_df.index.isin(val_edge_index_df.index)].reset_index(
        drop=True)
    train_edge_index_df.reset_index(drop=True, inplace=True)
    val_edge_index_df.reset_index(drop=True, inplace=True)

    num_nodes_set_one = train_edge_index_df['column_one'].nunique()
    num_nodes_set_two = train_edge_index_df['column_two'].nunique()

    node_one_train = train_edge_index_df['column_one'].unique().tolist()
    node_two_train = train_edge_index_df['column_two'].unique().tolist()

    val_edge_index_df = val_edge_index_df[val_edge_index_df['column_one'].isin(node_one_train)]
    val_edge_index_df = val_edge_index_df[val_edge_index_df['column_two'].isin(node_two_train)]
    test_edge_index_df = test_edge_index_df[test_edge_index_df['column_one'].isin(node_one_train)]
    test_edge_index_df = test_edge_index_df[test_edge_index_df['column_two'].isin(node_two_train)]

    if save_df:
        train_edge_index_df.to_csv(os.path.dirname(path) + '/train.tsv', sep='\t', header=None, index=None)
        val_edge_index_df.to_csv(os.path.dirname(path) + '/val.tsv', sep='\t', header=None, index=None)
        test_edge_index_df.to_csv(os.path.dirname(path) + '/test.tsv', sep='\t', header=None, index=None)

    node_one_public_private = {p: idx for idx, p in enumerate(node_one_train)}
    node_two_public_private = {d: idx for idx, d in enumerate(node_two_train)}
    node_one_private_public = {idx: p for idx, p in enumerate(node_one_train)}
    node_two_private_public = {idx: d for idx, d in enumerate(node_two_train)}

    train_rows = train_edge_index_df['column_one'].tolist()
    train_cols = train_edge_index_df['column_two'].tolist()

    val_rows = val_edge_index_df['column_one'].tolist()
    val_cols = val_edge_index_df['column_two'].tolist()

    test_rows = test_edge_index_df['column_one'].tolist()
    test_cols = test_edge_index_df['column_two'].tolist()

    train_edge_index_df = pd.concat([pd.Series(itemgetter(*train_rows)(node_one_public_private)),
                                     pd.Series(itemgetter(*train_cols)(node_two_public_private))], axis=1)

    val_edge_index_df = pd.concat([pd.Series(itemgetter(*val_rows)(node_one_public_private)),
                                   pd.Series(itemgetter(*val_cols)(node_two_public_private))], axis=1)

    test_edge_index_df = pd.concat([pd.Series(itemgetter(*test_rows)(node_one_public_private)),
                                    pd.Series(itemgetter(*test_cols)(node_two_public_private))], axis=1)

    neg_rows, neg_cols = np.where(1 - sp.coo_matrix((np.ones(len(train_rows)),
                                                     (train_edge_index_df[0].tolist(),
                                                      train_edge_index_df[1].tolist()))).todense() != 0)

    neg_edges = list(zip(neg_rows.tolist(), neg_cols.tolist()))
    random.shuffle(neg_edges)

    train_neg_edges = pd.DataFrame(neg_edges[:len(train_edge_index_df)])
    val_neg_edges = pd.DataFrame(neg_edges[len(train_edge_index_df):
                                           len(train_edge_index_df) +
                                           len(val_edge_index_df)])

    test_neg_edges = pd.DataFrame(neg_edges[len(train_edge_index_df) + len(val_edge_index_df):
                                            len(train_edge_index_df) +
                                            len(val_edge_index_df) +
                                            len(test_edge_index_df)])

    train_neg_edges_public = train_neg_edges.copy()
    val_neg_edges_public = val_neg_edges.copy()
    test_neg_edges_public = test_neg_edges.copy()

    train_neg_edges_public[0] = train_neg_edges_public[0].map(node_one_private_public)
    train_neg_edges_public[1] = train_neg_edges_public[1].map(node_two_private_public)

    val_neg_edges_public[0] = val_neg_edges_public[0].map(node_one_private_public)
    val_neg_edges_public[1] = val_neg_edges_public[1].map(node_two_private_public)

    test_neg_edges_public[0] = test_neg_edges_public[0].map(node_one_private_public)
    test_neg_edges_public[1] = test_neg_edges_public[1].map(node_two_private_public)

    if save_df:
        train_neg_edges_public.to_csv(os.path.dirname(path) + '/train_neg.tsv', sep='\t', header=None, index=None)
        val_neg_edges_public.to_csv(os.path.dirname(path) + '/val_neg.tsv', sep='\t', header=None, index=None)
        test_neg_edges_public.to_csv(os.path.dirname(path) + '/test_neg.tsv', sep='\t', header=None, index=None)

    train_rows = [node_one_public_private[r] for r in train_rows]
    train_cols = [node_two_public_private[c] + num_nodes_set_one for c in train_cols]

    edge_index = torch.tensor([train_rows + train_cols, train_cols + train_rows], dtype=torch.long)
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       sparse_sizes=(num_nodes_set_one + num_nodes_set_two,
                                     num_nodes_set_one + num_nodes_set_two))

    return train_edge_index_df, \
           val_edge_index_df, \
           test_edge_index_df, \
           train_neg_edges, \
           val_neg_edges, \
           test_neg_edges, \
           num_nodes_set_one, \
           num_nodes_set_two, \
           adj


def generate_datasets_hetero(data_path, train_edge_index_df, val_edge_index_df):
    df = pd.read_csv(data_path, sep='\t')
    df.columns = ['column_one', 'column_two']
    train_edge_index_df.columns = ['column_one', 'column_two']
    val_edge_index_df.columns = ['column_one', 'column_two']

    test_edge_index_df = df[~df.index.isin(pd.concat([train_edge_index_df,
                                                      val_edge_index_df], axis=0).index)].reset_index(drop=True)

    num_nodes_set_one = train_edge_index_df['column_one'].nunique()
    num_nodes_set_two = train_edge_index_df['column_two'].nunique()

    node_one_train = train_edge_index_df['column_one'].unique().tolist()
    node_two_train = train_edge_index_df['column_two'].unique().tolist()

    val_edge_index_df = val_edge_index_df[val_edge_index_df['column_one'].isin(node_one_train)]
    val_edge_index_df = val_edge_index_df[val_edge_index_df['column_two'].isin(node_two_train)]
    test_edge_index_df = test_edge_index_df[test_edge_index_df['column_one'].isin(node_one_train)]
    test_edge_index_df = test_edge_index_df[test_edge_index_df['column_two'].isin(node_two_train)]

    node_one_public_private = {p: idx for idx, p in enumerate(node_one_train)}
    node_two_public_private = {d: idx for idx, d in enumerate(node_two_train)}


    train_rows = train_edge_index_df['column_one'].tolist()
    train_cols = train_edge_index_df['column_two'].tolist()

    val_rows = val_edge_index_df['column_one'].tolist()
    val_cols = val_edge_index_df['column_two'].tolist()

    test_rows = test_edge_index_df['column_one'].tolist()
    test_cols = test_edge_index_df['column_two'].tolist()

    train_edge_index_df = pd.concat([pd.Series(itemgetter(*train_rows)(node_one_public_private)),
                                     pd.Series(itemgetter(*train_cols)(node_two_public_private))], axis=1)

    val_edge_index_df = pd.concat([pd.Series(itemgetter(*val_rows)(node_one_public_private)),
                                   pd.Series(itemgetter(*val_cols)(node_two_public_private))], axis=1)

    test_edge_index_df = pd.concat([pd.Series(itemgetter(*test_rows)(node_one_public_private)),
                                    pd.Series(itemgetter(*test_cols)(node_two_public_private))], axis=1)

    neg_rows, neg_cols = np.where(1 - sp.coo_matrix((np.ones(len(train_rows)), (train_edge_index_df[0].tolist(),
                                                                                train_edge_index_df[
                                                                                    1].tolist()))).todense() != 0)
    neg_edges = list(zip(neg_rows.tolist(), neg_cols.tolist()))
    random.shuffle(neg_edges)

    train_neg_edges = neg_edges[:len(train_edge_index_df)]
    val_neg_edges = neg_edges[len(train_edge_index_df): len(train_edge_index_df) + len(val_edge_index_df)]
    test_neg_edges = neg_edges[len(train_edge_index_df) + len(val_edge_index_df):
                               len(train_edge_index_df) + len(val_edge_index_df) + len(test_edge_index_df)]

    train_rows = [node_one_public_private[r] for r in train_rows]
    train_cols = [node_two_public_private[c] + num_nodes_set_one for c in train_cols]

    edge_index = torch.tensor([train_rows + train_cols, train_cols + train_rows], dtype=torch.long)
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       sparse_sizes=(num_nodes_set_one + num_nodes_set_two,
                                     num_nodes_set_one + num_nodes_set_two))

    return train_edge_index_df, \
           val_edge_index_df, \
           test_edge_index_df, \
           train_neg_edges, \
           val_neg_edges, \
           test_neg_edges, \
           num_nodes_set_one, \
           num_nodes_set_two, \
           adj


def get_patient_attributes():
    useful_attributes = ['sesso_1M', 'eta', 'PAS', 'PAD', 'alt_m', 'peso_kg', 'BMI', 
                         'Fumo', 'RBC', 'EMOGLOBINA', 'PLT', 'WBC', 'Insulina', 'HOMA', 
                         'GLUC', 'GOT', 'SGPT', 'γGT', 'colesterolo', 'HDL', 'LDL', 'TG',
                         'NAFLD']
#    useful_attributes = ['GENDER', 'AGE', 'GLU', 'HBA1 C', 'UREA', 'CREATININA', 'EGFR',
#                         'ACIDO URICO', 'GOT', 'GPT', 'GGT', 'PROTEINE TOTALI', 'SODIO',
#                         'POTASSIO', 'CALCIO', 'FOSFORO', 'MAGNESIO', 'COLESTEROLO',
#                         'HDL COLESTEROLO', 'LDL COLESTEROLO', 'TRIGLICERIDI', 'TSH', 'FT3',
#                         'FT4']

    patient_disease = pd.read_csv(f"data/patient_disease/dataset.tsv", sep='\t')
    patient_all_columns = pd.read_excel(f"data/database_modified.xlsx", header=0)
    patients_train = patient_disease["PATIENT_ID"]#.map(lambda x: x.split('_')[1])
    patients_train = pd.unique(patients_train).astype(int)
    mapping = {f"PATIENT_{k}": i for i, k in enumerate(patients_train)}
    patients_data = patient_all_columns[patient_all_columns['id'].isin(patients_train.astype(int))].sort_values('id')
    patients_data = patients_data[useful_attributes]
    patients_header = patients_data.columns.tolist()
    patients_data = patients_data.to_numpy()
    patients_data = np.where(np.isnan(patients_data), np.nanmean(patients_data, axis=0), patients_data)
    return normalize_patient_attributes(patients_data)[0], patients_header, mapping, normalize_patient_attributes(patients_data)[1]


def normalize_patient_attributes(data):
    scaler = StandardScaler()
#    data[:, 0] = data[:, 0] - 1
    scaler.fit(data)
#    scaler.mean_[0] = 0.5
#    scaler.var_[0] = 1
    data_norm = scaler.transform(data)
    # data_norm = np.concatenate([np.expand_dims(data[:, 0], -1), data_norm], axis=1)
    return data_norm, scaler


if __name__ == '__main__':
    g_train, g_ind = generate_bipartite_datasets_for_inductive("patient_disease", 100)
