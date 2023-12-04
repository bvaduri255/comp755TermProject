import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Fit new model to input features and labels, return model & model parameters
def fit_model(data):
    features, labels = data[:, :-1], data[:, -1:]
    labels = labels.ravel()
    model = LogisticRegression(solver="lbfgs", max_iter=100000)
    model.fit(features, labels)
    coef = model.coef_
    intercept = model.intercept_
    classes = model.classes_
    return model, coef, intercept, classes


# Load parameter into model, return model
def load_model(agg_coef, agg_intercept, agg_classes):
    model = LogisticRegression(solver="lbfgs", max_iter=100000)
    model.coef_ = agg_coef
    model.intercept_ = agg_intercept
    model.classes_ = agg_classes
    return model


# Predict on new data, return metrics (accuracy, confusion matrix, AUC)
def eval_model(model, test_data):
    features, labels = test_data[:, :-1], test_data[:, -1:]
    labels = labels.ravel()
    model_acc = model.score(features, labels)
    probas = model.predict_proba(features)[:, 1]
    try:
        model_auc = roc_auc_score(labels, probas)
    except ValueError:
        model_auc = None
    return model_acc, model_auc

def aggregate_classes(classes):
    return classes[0]


if __name__ == "__main__":
    from run_model import read_data, combine_split_datasets
    DATA_FOLDER = "heart_data"

    train_splits, eval_splits, test_splits = read_data(DATA_FOLDER)
    full_train, full_eval, full_test = combine_split_datasets(train_splits, eval_splits, test_splits)

    X_train, y_train = full_train[:, :-1], full_train[:, -1:]
    X_test, y_test = full_test[:, :-1], full_test[:, -1:]

    model, *params = fit_model(full_train)
    accuracy, auc = eval_model(model, full_test)
    print(f"Baseline model: Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

    # Load model faithfulness
    loaded_model = load_model(*params)
    accuracy, auc = eval_model(loaded_model, full_test)
    print(f"Loaded model: Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")