import numpy as np
import xgboost as xgb
from thesis_dataset import create_centralized_dataset, ThesisDataset

params = {
    "objective": "multi:softmax",
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
    'num_class': 3,
}

# start learning after 1.0 / learning rate


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def softmax_obj(preds, dtrain: xgb.DMatrix):
    labels = dtrain.get_label()
    p = softmax(preds)
    wt = dtrain.get_weight()
    eps = 1e-16
    hess = np.max(2.0 * p * (1.0 - p) * wt, eps)

    return p * wt, hess   

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

    # Right now (XGBoost 1.0.0), reshaping is necessary
    return grad, hess

def gradient_based_one_side_sampling(a = 0.05, b = 0.05, ):
    train_dmatrix, test_dmatrix = create_centralized_dataset(ThesisDataset.IRIS.value)
    
    bst = xgb.Booster(params, [train_dmatrix])

    # bst = xgb.train(
    #     params,
    #     xgb.DMatrix(empty),
    #     num_boost_round=1,
    #     evals=[(test_dmatrix, "test"), (train_dmatrix, "train")],
    #     #feval try
    # )

    fact = (1 - a) / b
    topN = a * train_dmatrix.num_row()
    randN = b * train_dmatrix.num_row()
    
    preds = bst.predict(train_dmatrix, output_margin=True, training=True)
    print(preds.shape)

    # loss function TBD
    # gradients, hessians = softmax_obj(preds, train_dmatrix)
    gradients, hessians = softprob_obj(preds, train_dmatrix)

    weights = np.ones_like(train_dmatrix.get_label())
    sorted_indices = np.argsort(np.abs(np.sum(gradients, axis=1, keepdims=False)))
    topSet = sorted_indices[:int(topN)]
    randSet = np.random.choice(sorted_indices[int(topN):], int(randN), replace=False)
    usedSet = np.concatenate([topSet, randSet])
    weights[randSet] *= fact
    new_train_dmatrix = train_dmatrix.slice(usedSet)
    new_train_dmatrix.set_weight(weights[usedSet])
    
    # bst = xgb.Booster(params, [new_train_dmatrix])
    # bst.boost(new_train_dmatrix, gradients[usedSet], hessians[usedSet])
    
    bst = xgb.train(
       params,
       new_train_dmatrix,
       num_boost_round=1,
       evals=[(test_dmatrix, "test")],
       #feval try
    )
    
   # print(bst.eval_set([(test_dmatrix, "test")]))
    
    # information gain, gradients and hessians
    # idea: use divergence score federated vs centralized
    # invite William to repo


    models = []    
    return models

def my_function(data: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
    # Some operations that modify dmatrix
    dmatrix.get_data()
    dmatrix = xgb.DMatrix(data=data)
    # Do something with new_dmatrix
    return np.array([1, 2, 3]), np.array([1, 2, 3])

# Outside the function
data = np.array([[1, 2], [3, 4]])
original_dmatrix = xgb.DMatrix(data=data)

result_array, modified_dmatrix = my_function(data, original_dmatrix)

gradient_based_one_side_sampling()