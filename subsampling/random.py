import numpy as np
import xgboost as xgb
from subsampling.subsampling_strategy import SubsamplingStrategy
from visualization import plot_tree, plot_labels
from flwr_datasets.partitioner import IidPartitioner
from typing import Optional, Tuple


# start learning after 1.0 / learning rate

def gradient_based_one_side_sampling(dataset: str = "", a = 0.05, b = 0.05, ):
    dataset = WineQualityDataloader(IidPartitioner(3))
    train_dmatrix, _ = dataset.get_train_dmatrix()
    print(train_dmatrix.get_label())
    test_dmatrix, _ = dataset.get_test_dmatrix(None)
    print(test_dmatrix.get_label())
    
    bst = xgb.Booster(params, [train_dmatrix])

    fact = (1 - a) / b
    topN = a * train_dmatrix.num_row()
    randN = b * train_dmatrix.num_row()
    for i in range(5):
        preds = bst.predict(train_dmatrix, output_margin=True, training=True)

        gradients, _ = rmse_obj(preds, train_dmatrix)

        weights = np.ones_like(train_dmatrix.get_label())
        print(gradients)
        sorted_indices = np.argsort(np.abs(np.sum(gradients, axis=1, keepdims=False)))
        topSet = sorted_indices[:int(topN)]
        randSet = np.random.choice(sorted_indices[int(topN):], int(randN), replace=False)
        usedSet = np.concatenate([topSet, randSet])
        weights[randSet] *= fact
        new_train_dmatrix = train_dmatrix.slice(usedSet)
        new_train_dmatrix.set_weight(weights[usedSet])

        bst.update(new_train_dmatrix, 1)
        print(bst.eval_set([(test_dmatrix, "test")]))

        plot_tree(bst)
        plot_labels(3, IrisDataloader(IidPartitioner(3)), MVS(rmse_obj), bst, i)
    return bst

def run_centralized():
    dataset = WineQualityDataloader(IidPartitioner(20))
    train_dmatrix, _ = dataset.get_train_dmatrix()
    print(train_dmatrix.get_label())
    test_dmatrix, _ = dataset.get_test_dmatrix(None)
    print(test_dmatrix.get_label())
    #bst = xgb.Booster(params, [train_dmatrix])
    #bst.update(train_dmatrix, 1)
    bst = xgb.train(params=params, dtrain=train_dmatrix, num_boost_round=1, evals={(test_dmatrix, "valid")})
    
    print(bst.eval_set([(test_dmatrix, "test")]))

# gradient_based_one_side_sampling()

class GOSS(SubsamplingStrategy):

    def __init__(self, objective, a = 0.05, b = 0.05) -> None:
        self.objective = objective
        self.a = a
        self.b = b
        self.regularized_gradients_cache: Optional[np.ndarray] = None
        self.threshold = []

    def threshold_subsample(self, train_dmatrix: xgb.DMatrix, threshold: int) -> xgb.DMatrix:
        if self.regularized_gradients_cache is None:
            raise ValueError("No regularized gradients in memory.")
        
        indices = np.where(self.regularized_gradients_cache >= threshold)[0]
        new_train_dmatrix = train_dmatrix.slice(indices)
        return new_train_dmatrix

    def subsample(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> xgb.DMatrix:
        fact = (1 - self.a) / self.b
        topN = self.a * train_dmatrix.num_row()
        randN = self.b * train_dmatrix.num_row()

        gradients, _ = self.objective(predictions, train_dmatrix)

        weights = np.ones_like(train_dmatrix.get_label())
        sorted_indices = np.argsort(np.abs(np.sum(gradients, axis=1, keepdims=False) if gradients.ndim > 1 else gradients))
        topSet = sorted_indices[:int(topN)]
        randSet = np.random.choice(sorted_indices[int(topN):], int(randN), replace=False)
        usedSet = np.concatenate([topSet, randSet])
        weights[randSet] *= fact
        new_train_dmatrix = train_dmatrix.slice(usedSet)
        new_train_dmatrix.set_weight(weights[usedSet])

        return new_train_dmatrix
    
    def subsample_indices(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix, grad_hess: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        pass

    def grad_and_hess(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    def global_sampling(self, grad_hess_dict: dict[int, list[(float, float)]]) -> dict[int, list[int]]:
        sampling_values = {}
        all_gradients = []
        all_hessians = []

        for id, (grad, hess) in grad_hess_dict.items():
            all_gradients.extend(grad)
            all_hessians.extend(hess)

        all_gradients = np.array(all_gradients)
        all_hessians = np.array(all_hessians)

    def get_threshold(self) -> int:
        return self.threshold[-1]