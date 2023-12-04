import os
import pandas as pd
import numpy as np
from pathlib import Path

def read_data(folder_name):
    """Given name of dataset folder, read all data from splits into lists"""
    train_splits, eval_splits, test_splits = [], [], []

    data_path = os.path.join(os.getcwd(), Path(folder_name))
    files = os.listdir(data_path)
    num_splits = len(files) // 3
    print(f"Calculated number of splits in dataset: {num_splits}")

    for split_i in range(num_splits):
        train_file = os.path.join(data_path, f"split{split_i}_train")
        eval_file = os.path.join(data_path, f"split{split_i}_eval")
        test_file = os.path.join(data_path, f"split{split_i}_test")

        train_split = np.genfromtxt(train_file, delimiter=",")
        eval_split = np.genfromtxt(eval_file, delimiter=",")
        test_split = np.genfromtxt(test_file, delimiter=",")

        train_splits.append(train_split)
        eval_splits.append(eval_split)
        test_splits.append(test_split)
    return train_splits, eval_splits, test_splits
        

def combine_split_datasets(train_splits, eval_splits, test_splits):
    train_combined = np.vstack(train_splits)
    eval_combined = np.vstack(eval_splits)
    test_combined = np.vstack(test_splits)
    return train_combined, eval_combined, test_combined


def get_agg_model(model_type, agg_type, train_splits, eval_splits, test_splits):
    """Fits split models, returns split performance metrics and aggregate model"""
    # Determine model and parameters to aggregate
    if model_type == "logistic regression":
        from logistic_regression import fit_model, load_model, eval_model, aggregate_classes
        coefs, intercepts, classes = [], [], []
        parameters = [coefs, intercepts, classes]

        # Dictionary denote custom aggregation functions specific to the model. {index: function}
        custom_agg_funcs = {2: aggregate_classes}

    elif model_type == "neural network":
        from neural_net import fit_model, load_model, eval_model
        l1_w, l2_w, l3_w, l4_w = [], [], [], []
        l1_b, l2_b, l3_b, l4_b = [], [], [], []
        parameters = [l1_w, l1_b, l2_w, l2_b, l3_w, l3_b, l4_w, l4_b]
        custom_agg_funcs = {}


    # Determine aggregation method
    if agg_type == "simple mean":
        from agg_functions import simple_mean
        agg_func = simple_mean

    # Train each split model, keep track of each splits performance, and fit & evalutate aggregate model
    split_accs, split_aucs = [], []
    for split_i in range(len(train_splits)):
        print(f"Fitting split {split_i}")
        train_data = train_splits[split_i]
        eval_data = eval_splits[split_i]
        test_data = test_splits[split_i]

        model, *learned_params = fit_model(train_data)
        assert len(parameters) == len(learned_params)
        for j in range(len(parameters)):
            parameters[j].append(learned_params[j])
        split_acc, split_auc = eval_model(model, test_data)
        split_accs.append(split_acc)
        split_aucs.append(split_auc)
    
    print("Fitting aggregate model")
    agg_parameters = []
    for i in range(len(parameters)):
        if i in custom_agg_funcs:
            agg_parameters.append(custom_agg_funcs[i](parameters[i]))
        else:
            agg_parameters.append(agg_func(parameters[i]))
    agg_model = load_model(*agg_parameters)
    full_test_data = np.vstack(test_splits)
    agg_acc, agg_auc = eval_model(agg_model, full_test_data)

    return split_accs, split_aucs, agg_acc, agg_auc


if __name__ == "__main__":
    DATA_FOLDER = "heart_data"
    train_splits, eval_splits, test_splits = read_data(DATA_FOLDER)
    full_train, full_eval, full_test = combine_split_datasets(train_splits, eval_splits, test_splits)

    MODEL_TYPE = "logistic regression"
    AGG_TYPE = "simple mean"
    split_accs, split_aucs, agg_acc, agg_auc = get_agg_model(MODEL_TYPE, AGG_TYPE, train_splits, eval_splits, test_splits)

    for i in range(len(split_accs)):
        print(f"Split {i} has acc {split_accs[i]} and auc {split_aucs[i]}")

    print(f"Aggregate model has acc: {agg_acc} and AUC: {agg_auc}")
    print(f"The mean of split accs: {np.mean(split_accs)}. The mean of split AUCs: {np.mean(split_aucs)}")
