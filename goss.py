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
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def softmax_obj(preds, dtrain: xgb.DMatrix):
    labels = dtrain.get_label()
    p = softmax(preds)
    wt = dtrain.get_weight()
    eps = 1e-16
    hess = np.max(2.0 * p * (1.0 - p) * wt, eps)

    return p * wt, hess   

def gradient_based_one_side_sampling(a = 0.05, b = 0.05, ):
    train_dmatrix, test_dmatrix = create_centralized_dataset(ThesisDataset.IRIS.value)
    empty = np.array([[0.0] * 4])
   
    bst = xgb.train(
        params,
        xgb.DMatrix(empty),
        num_boost_round=1,
        evals=[(test_dmatrix, "test"), (train_dmatrix, "train")],
        #feval try
    )

    fact = (1 - a) / b
    topN = a * train_dmatrix.num_row()
    randN = b * train_dmatrix.num_row()
    
    preds = bst.predict(train_dmatrix, output_margin=True)
    print(preds)
    print(train_dmatrix.get_label()[0])
    print(preds[0])
    print(softmax(preds))

    # loss function TBD
    # gradients, hessians = softmax_obj(preds, train_dmatrix)
    gradients = np.abs(train_dmatrix.get_label() - preds)

    weights = np.ones_like(gradients)
    sorted_indices = np.argsort(np.abs(gradients))
    topSet = sorted_indices[:int(topN)]
    randSet = np.random.choice(sorted_indices[int(topN):], int(randN), replace=False)
    usedSet = np.concatenate([topSet, randSet])
    weights[randSet] *= fact
    new = train_dmatrix.slice(usedSet)
    new.set_weight(weights[usedSet])
    
    bst = xgb.train(
        params,
        new,
        num_boost_round=1,
        evals=[(test_dmatrix, "test"), (train_dmatrix, "train")],
        #feval try
    )
    
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