from itertools import product
import importlib
import argparse
from datetime import datetime
import copy

from utils.grid_search_parameters import (
    model_parameters_gnn,
    random_seed,
    epochs,
    patience,
)
from utils.results import *
from utils.dataset import *
from sklearn.model_selection import ParameterGrid

model_parameters = model_parameters_gnn

parser = argparse.ArgumentParser(description="Run link prediction gnn.")
parser.add_argument("--dataset", type=str, default="patient_disease")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--save_df", type=bool, default=True)
parser.add_argument("--unseen_ratio", type=float, default=0)
parser.add_argument(
    "--force_use_patient_features", type=bool, default=False
)  # this has to be set if we want to use patient features in the transductive setting

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

batch_size = args.batch_size
adj_extended = None

if args.dataset == "patient_disease":  # TODO: for now the only inductive graph
    (
        g,
        g_ind,
        num_nodes_set_one,
        num_nodes_set_two,
        adj,
        patient_attributes,
        nodes_set_one,
        nodes_set_two,
    ) = generate_bipartite_datasets_for_inductive(
        "patient_disease",
        random_seed,
        unseen_ratio=args.unseen_ratio,
        save_df=args.save_df,
    )
    if args.unseen_ratio > 0:
        train_df = g[0]
        train_neg_edges = g[3]
        val_df = g[2]
        val_neg_edges = g[5]
        test_df = g_ind[2]
        test_neg_edges = g_ind[5]
        adj, adj_extended = adj
        emb_set_node_one = torch.tensor(np.array(patient_attributes))
    else:
        train_df = g[0]
        train_neg_edges = g[3]
        val_df = g[1]
        val_neg_edges = g[4]
        test_df = g[2]
        test_neg_edges = g[5]
        adj = adj
        emb_set_node_one = None

else:
    (
        train_df,
        val_df,
        test_df,
        train_neg_edges,
        val_neg_edges,
        test_neg_edges,
        num_nodes_set_one,
        num_nodes_set_two,
        adj,
    ) = generate_datasets(args.dataset, random_seed, save_df=args.save_df)

for key1, value1 in model_parameters.items():
    print("****************************************")
    print(f"CURRENT MODEL: {key1}")

    model_class = getattr(importlib.import_module(f"models.{key1.lower()}"), f"{key1}")

    print("All parameters:")
    for key2, value2 in value1.items():
        print(f"- {key2}: {value2}")

    # all_configurations = list(product(*list(value1.values())))
    all_configurations = list(ParameterGrid(value1))

    best_validation = -np.inf
    best_configuration = 1
    best_model = None
    test_metrics = []
    losses_dict = dict()

    emb_set_node_one = None
    if args.dataset == "patient_disease" and args.force_use_patient_features:
        emb_set_node_one = torch.tensor(np.array(patient_attributes))
    for idx, conf in enumerate(all_configurations):
        print(f"\n\nHyper-parameter configuration: {idx + 1}, with parameters: {conf}")
        model = model_class(
            num_set_node_one=num_nodes_set_one,
            num_set_node_two=num_nodes_set_two,
            adj=adj,
            random_seed=random_seed,
            conf=conf,
            emb_set_node_one=emb_set_node_one,
            adj_extended=adj_extended,
        )

        (
            current_best_validation_accuracy,
            current_all_train_accuracies,
            current_all_val_accuracies,
            current_all_train_losses,
            current_all_val_losses,
            elapsed_time,
        ) = model.fit(
            epochs=epochs,
            num_interactions=int(len(train_df) // batch_size),
            patience=patience,
            train_edge_index_df=train_df,
            train_neg_edges=train_neg_edges,
            batch_size=batch_size,
            val_edge_index_df=val_df,
            val_neg_edges=val_neg_edges,
        )
        losses_dict[get_config_name(list(value1.keys()), conf)] = {
            "train": current_all_train_losses,
            "val": current_all_val_losses,
            "train_acc": current_all_train_accuracies,
            "val_acc": current_all_val_accuracies,
        }
        test_dict = model.predict(
            test_df.to_numpy(), np.array(test_neg_edges), validation=False
        )
        test_dict.update({"training_time": elapsed_time})
        print("EVALUATION ON TEST SET...")
        print(
            f'Test accuracy: {test_dict["accuracy"]:.3f}\n'
            f'Test recall_0: {test_dict["recall_0"]:.3f}\n'
            f'Test precision_0: {test_dict["precision_0"]:.3f}\n'
            f'Test f1_score_0: {test_dict["f1_score_0"]:.3f}\n'
            f'Test recall_1: {test_dict["recall_1"]:.3f}\n'
            f'Test precision_1: {test_dict["precision_1"]:.3f}\n'
            f'Test f1_score_1: {test_dict["f1_score_1"]:.3f}\n'
            f'Test auc: {test_dict["auc"]:.3f}\n'
            f'True negative: {test_dict["tn"]}\n'
            f'False positive: {test_dict["fp"]}\n'
            f'False negative: {test_dict["fn"]}\n'
            f'True positive: {test_dict["tp"]}\n'
        )
        test_metrics.append(test_dict)
        save_results(
            results_dict=test_dict,
            dataset=args.dataset,
            model_name=key1,
            name_params=list(value1.keys()),
            config=conf,
            name=timestamp_str,
        )

        if best_validation < current_best_validation_accuracy:
            best_validation = current_best_validation_accuracy
            best_configuration = idx + 1
            best_model = copy.deepcopy(model)


    print(
        f"\n\nEnd hyper-parameter tuning for {key1}. Best configuration: {best_configuration}"
    )
    print(
        f"Hyper-parameters best configuration: {all_configurations[best_configuration - 1]}\n"
    )
    print(f"Test metrics:")
    print(
        f'Test accuracy: {test_metrics[best_configuration - 1]["accuracy"]:.3f}\n'
        f'Test recall_0: {test_metrics[best_configuration - 1]["recall_0"]:.3f}\n'
        f'Test precision_0: {test_metrics[best_configuration - 1]["precision_0"]:.3f}\n'
        f'Test f1_score_0: {test_metrics[best_configuration - 1]["f1_score_0"]:.3f}\n'
        f'Test recall_1: {test_metrics[best_configuration - 1]["recall_1"]:.3f}\n'
        f'Test precision_1: {test_metrics[best_configuration - 1]["precision_1"]:.3f}\n'
        f'Test f1_score_1: {test_metrics[best_configuration - 1]["f1_score_1"]:.3f}\n'
        f'Test auc: {test_metrics[best_configuration - 1]["auc"]:.3f}\n'
        f'True negative: {test_metrics[best_configuration - 1]["tn"]}\n'
        f'False positive: {test_metrics[best_configuration - 1]["fp"]}\n'
        f'False negative: {test_metrics[best_configuration - 1]["fn"]}\n'
        f'True positive: {test_metrics[best_configuration - 1]["tp"]}\n'
    )
    save_results_best(
        results_dict=test_metrics[best_configuration - 1],
        dataset=args.dataset,
        best_model=best_model,
        model_name=key1,
        name_params=list(value1.keys()),
        config=all_configurations[best_configuration - 1],
        name=timestamp_str,
    )
    save_losses_best(
        losses=losses_dict[
            get_config_name(
                list(value1.keys()), all_configurations[best_configuration - 1]
            )
        ],
        dataset=args.dataset,
        model_name=key1,
        name=timestamp_str,
    )
