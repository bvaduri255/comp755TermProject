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
    model_auc = roc_auc_score(labels, probas)
    return model_acc, model_auc

def aggregate_classes(classes):
    return classes[0]