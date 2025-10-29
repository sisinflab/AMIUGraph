from itertools import product
import importlib
import argparse
from datetime import datetime
import copy

from utils.grid_search_parameters import (
    model_parameters_kge,
    random_seed,
    epochs,
    patience,
)
from utils.results import *
from utils.dataset import *


model_parameters = model_parameters_kge

parser = argparse.ArgumentParser(description="Run link prediction kge.")
# parser.add_argument('--all_datasets', type=list, default=['patient_disease',
#                                                          'drug_disease',
#                                                          'drug_target'])


parser.add_argument("--dataset", type=str, default="patient_disease")
#parser.add_argument("--all_datasets", type=list, default=["patient_disease"])
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--optimizer", type=str, default="Adam") # Adam
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--save_df", type=bool, default=True)
parser.add_argument("--unseen_ratio", type=float, default=0)  # 0 for transductive
parser.add_argument(
    "--force_use_patient_features", type=bool, default=False
)  # this has to be set if we want to use patient features in the transductive setting

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
batch_size = args.batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

args.all_datasets = [args.dataset]


if args.all_datasets == ["patient_disease"]:  # TODO: for now the only inductive graph
    (
        g,
        g_ind,
        num_nodes_set_one,
        num_nodes_set_two,
        _,
        patient_attributes,
        nodes_set_one,
        nodes_set_two,
    ) = generate_bipartite_datasets_for_inductive(
        "patient_disease",
        random_seed,
        unseen_ratio=args.unseen_ratio,
        save_df=args.save_df,
    )
    num_nodes_train = num_nodes_set_one + num_nodes_set_two
    num_relations = 1
    relations_labels = ["patient_disease"]

    if args.unseen_ratio > 0:

        # if unseen ratio is higher than 0
        # the setting is inductive
        # therefore, patients will be initialized with their numerical features

        train_df = g[0]
        train_neg_edges = g[3]
        val_df = g[2]
        val_neg_edges = g[5]
        test_df = g_ind[2]
        test_neg_edges = g_ind[5]
        inductive_ids = list(range(len(nodes_set_one)))
        inductive_embs = patient_attributes
    else:
        # else, the setting is TRANSDUCTIVE
        # therefore, patients are initialized randomly

        train_df = g[0]
        train_neg_edges = g[3]
        val_df = g[1]
        val_neg_edges = g[4]
        test_df = g[2]
        test_neg_edges = g[5]
        inductive_ids = []
        inductive_embs = None

    if args.force_use_patient_features and args.all_datasets == ["patient_disease"]:
        inductive_ids = list(range(len(nodes_set_one)))
        inductive_embs = patient_attributes

    train_df[1] += num_nodes_set_one
    val_df[1] += num_nodes_set_one
    test_df[1] += num_nodes_set_one

    train_df[2] = 0
    train_neg_edges[2] = 0
    val_df[2] = 0
    val_neg_edges[2] = 0
    test_df[2] = 0
    test_neg_edges[2] = 0

else:
    (
        train_df,
        val_df,
        test_df,
        train_neg_edges,
        val_neg_edges,
        test_neg_edges,
        num_nodes_train,
        num_relations,
        relations_labels,
    ) = generate_datasets_kg(args.all_datasets, random_seed)

    inductive_ids = []
    inductive_embs = None

for key1, value1 in model_parameters.items():
    print("****************************************")
    print(f"CURRENT MODEL: {key1}")

    model_class = getattr(
        importlib.import_module(f"models.kge.{key1.lower()}"), f"{key1}"
    )

    print("All parameters:")

    config_keys = []
    for key2, value2 in value1.items():
        print(f"- {key2}: {value2}")
        config_keys.append(key2)
    all_configurations = list(product(*list(value1.values())))

    best_validation = -np.inf
    best_configuration = 1
    best_model = None
    test_metrics = []
    losses_dict = dict()

    for idx, conf in enumerate(all_configurations):
        print(f"\n\nHyper-parameter configuration: {idx + 1}, with parameters: {conf}")
        model = model_class(
            num_set_node=num_nodes_train,
            relations=relations_labels,
            random_seed=random_seed,
            optimizer=args.optimizer,
            conf=dict(zip(config_keys, conf)),
            is_inductive=args.all_datasets == ["patient_disease"]
            and args.unseen_ratio > 0,
            inductive_ids=inductive_ids,
            inductive_embs=inductive_embs,
        ).to(device)
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

        print_results(test_dict, relations_labels)

        test_metrics.append(test_dict)
        save_results(
            results_dict=test_dict,
            dataset="_".join(args.all_datasets),
            model_name=key1,
            name_params=list(value1.keys()),
            config=conf,
            optimizer=args.optimizer,
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

    print_results(test_metrics[best_configuration - 1], relations_labels)

    save_results_best(
        results_dict=test_metrics[best_configuration - 1],
        dataset="_".join(args.all_datasets),
        best_model=best_model,
        model_name=key1,
        name_params=list(value1.keys()),
        config=all_configurations[best_configuration - 1],
        optimizer=args.optimizer,
        name=timestamp_str,
    )

    save_losses_best(
        losses=losses_dict[
            get_config_name(
                list(value1.keys()), all_configurations[best_configuration - 1]
            )
        ],
        dataset="_".join(args.all_datasets),
        model_name=key1,
        optimizer=args.optimizer,
        name=timestamp_str,
    )
