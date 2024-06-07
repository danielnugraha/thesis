import numpy as np
import xgboost as xgb
import math

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)
 

def softprob_obj(predt: np.ndarray, data: xgb.DMatrix):
    '''Loss function.  Computing the gradient and approximated hessian (diagonal).
    Reimplements the `multi:softprob` inside XGBoost.

    '''
    labels = data.get_label()
    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones_like(labels)
    else:
        weights = data.get_weight()

    # The prediction is of shape (rows, classes), each element in a row
    # represents a raw prediction (leaf weight, hasn't gone through softmax
    # yet).  In XGBoost 1.0.0, the prediction is transformed by a softmax
    # function, fixed in later versions.
    grad = np.zeros_like(predt)
    hess = np.zeros_like(predt)

    eps = 1e-6

    # compute the gradient and hessian, slow iterations in Python, only
    # suitable for demo.  Also the one in native XGBoost core is more robust to
    # numeric overflow as we don't do anything to mitigate the `exp` in
    # `softmax` here.
    for r in range(predt.shape[0]):
        target = labels[r]
        p = softmax(predt[r, :])
        for c in range(predt.shape[1]):

            g = p[c] - 1.0 if c == target else p[c]
            g = g * weights[r]
            h = max((2.0 * p[c] * (1.0 - p[c]) * weights[r]).item(), eps)
            grad[r, c] = g
            hess[r, c] = h

    return grad, hess


def binary_obj(predt: np.ndarray, data: xgb.DMatrix):
    labels = data.get_label()
    eps = 1e-6

    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones_like(labels)
    else:
        weights = data.get_weight()
    grad = predt - labels
    hess = np.maximum(predt * (1.0 - predt), eps)

    return grad * weights, hess * weights


def rmse_obj(predt: np.ndarray, data: xgb.DMatrix):
    labels = data.get_label()
    
    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones_like(labels)
    else:
        weights = data.get_weight()

    grad = predt - labels
    hess = 1.0
    return grad * weights, hess * weights


def rmsle_obj(predt: np.ndarray, data: xgb.DMatrix):
    labels = data.get_label()
    
    if data.get_weight().size == 0:
        # Use 1 as weight if we don't have custom weight.
        weights = np.ones_like(labels)
    else:
        weights = data.get_weight()

    predt = np.maximum(predt, -1 + 1e-6)
    grad = (np.log1p(predt) - np.log1p(labels)) / (predt + 1)
    hess = np.maximum((-np.log1p(predt) + np.log1p(labels) + 1) / ((predt + 1) ** 2), 1e-6)
    return grad * weights, hess * weights

def absolute_error_obj(predt: np.ndarray, data: xgb.DMatrix):
    x = predt - labels
    grad = np.sign(x) 
    grad[np.abs(x) < alpha] = 2/alpha*x[np.abs(x) < alpha]
    hess = np.zeros_like(labels) 
    hess[np.abs(x) < alpha] = 2/alpha
    return grad, hess