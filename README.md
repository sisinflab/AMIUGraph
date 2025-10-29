# AMIUGraph

This is the official repository for the paper _"AMIUGraph: Analysis and Modeling of Interactions for Utility-driven
Benchmarking of Graph-Based Models in Healthcare"_, under review in Expert Systems with Applications.

### Requirements

Before running any scripts, please make sure to install the proper Python packages by doing:

```sh
pip install -r requirements.txt
```

### Run the codes

Once you installed everything, you are ready to go! 

First, run the KGE models for link prediction:

```sh
python run_link_prediction_kge.py \
      --dataset <dataset_name> \
      --batch_size <batch_size> \
      --optimizer <optimizer_name> \
      --gpu <gpu_id> \ 
      --save_df <save_dataset_dataframe> \
      --unseen_ratio <ratio_for_inductive> \
      --force_use_patient_features <use_patient_features> \
```

For ```--unseen_ratio```, set it to 0.0 if you want to use the transductive setting, otherwise the ratio must be > 0.0. 

For ```--force_user_patient_features```, set it to True if you want to use patients features for the transductive setting.

Then, run the GNN models for link prediction:

```sh
python run_link_prediction_gnn.py \
      --dataset <dataset_name> \
      --batch_size <batch_size> \
      --gpu <gpu_id> \ 
      --save_df <save_dataset_dataframe> \
      --unseen_ratio <ratio_for_inductive> \
      --force_use_patient_features <use_patient_features> \
```

It has the exact same meaning as the code for KGE.
