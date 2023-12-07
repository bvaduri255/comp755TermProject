import os
import pandas as pd
import numpy as np
from pathlib import Path

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from art.utils import load_mnist
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

from tqdm import tqdm


from mnist_nn2 import *

def combine_split_datasets(train_splits, eval_splits, test_splits):
    train_combined = np.vstack(train_splits)
    eval_combined = np.vstack(eval_splits)
    test_combined = np.vstack(test_splits)
    return train_combined, eval_combined, test_combined


def get_agg_model(model_type, agg_type, x_train_splits, y_train_splits, x_test_splits, y_test_splits):
    """Fits split models, returns split performance metrics and aggregate model"""

    if model_type == "mnist nn":
        from mnist_nn import fit_model, load_model, eval_model
        l0_w, l2_w, l5_w, l6_w = [], [], [], []
        l0_b, l2_b, l5_b, l6_b = [], [], [], []
        parameters = [l0_w, l0_b, l2_w, l2_b, l5_w, l5_b, l6_w, l6_b]
        custom_agg_funcs = {}

    # Determine aggregation method
    if agg_type == "simple mean":
        from agg_functions import simple_mean
        agg_func = simple_mean
    if agg_type == "proximal operator weighted average":
        from agg_functions import proximal_operator_weighted_average
        agg_func = proximal_operator_weighted_average

    # Train each split model, keep track of each splits performance, and fit & evalutate aggregate model
    split_accs, split_aucs = [], []
    for split_i in range(len(x_train_splits)):
        print(f"Fitting split {split_i}")
        x_train = x_train_splits[split_i]
        y_train = y_train_splits[split_i]
        x_test = x_test_splits[split_i]
        y_test = y_test_splits[split_i]

        model, *learned_params = fit_model(x_train, y_train)
        assert len(parameters) == len(learned_params)
        for j in range(len(parameters)):
            parameters[j].append(learned_params[j])
        split_acc, split_auc = eval_model(model, x_test, y_test)
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
    agg_acc, agg_auc = eval_model(agg_model, x_test, y_test)

    return split_accs, split_aucs, agg_acc, agg_auc

def split_data(data, num_splits):
    N = len(data)
    split_size = N // num_splits
    last_split_size = split_size + N % num_splits
    splits = []

    for i in range(num_splits-1):
        start_idx, end_idx = i, i+split_size
        current_split = data[start_idx:end_idx, :]
        splits.append(current_split)
    splits.append(data[-last_split_size:, :])
    return splits

if __name__ == "__main__":
    """No longer using heart or shopping data"""
    # DATA_FOLDER = "heart_data"
    # train_splits, eval_splits, test_splits = read_data(DATA_FOLDER)
    # full_train, full_eval, full_test = combine_split_datasets(train_splits, eval_splits, test_splits)

    """Load MNIST dataset"""
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    # Split MNIST
    num_splits = 4
    x_train_splits = split_data(x_train, num_splits)
    y_train_splits = split_data(y_train, num_splits)
    x_test_splits = split_data(x_test, num_splits)
    y_test_splits = split_data(y_test, num_splits)

    # Create four nodes
    split_models = [create_model() for _ in range(num_splits)]

    # Run a round
    num_rounds = 300
    for round in tqdm(range(num_rounds)):
        split_gradients = []
        


    MODEL_TYPE = "mnist nn"     
    AGG_TYPE = "proximal operator weighted average"
    split_accs, split_aucs, agg_acc, agg_auc = get_agg_model(MODEL_TYPE, AGG_TYPE, x_train_splits, y_train_splits, x_test_splits, y_test_splits)

    for i in range(len(split_accs)):
        print(f"Split {i} has acc {split_accs[i]} and auc {split_aucs[i]}")

    print(f"Aggregate model has acc: {agg_acc} and AUC: {agg_auc}")
    print(f"The mean of split accs: {np.mean(split_accs)}. The mean of split AUCs: {np.mean(split_aucs)}")
