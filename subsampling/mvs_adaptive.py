from numpy import ndarray
from mvs import MVS
import xgboost as xgb
import numpy as np
from typing import List, Optional
from flwr_datasets.partitioner import IidPartitioner
from wine_quality_dataloader import WineQualityDataloader
from visualization import plot_tree, plot_labels
import matplotlib.pyplot as plt
from utils import params, softprob_obj, rmse_obj

class MVSAdaptive(MVS):

    def __init__(self, objective, lastIterTree: List[str], dimension: int, lambda_rate = 0.1, sample_rate = 0.1) -> None:
        super().__init__(objective, lambda_rate, sample_rate)
        self.lastIterTree = lastIterTree
        self.dimension = dimension

    def subsample(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> xgb.DMatrix:
        subsample = self.subsample_indices(predictions, train_dmatrix)

        new_train_dmatrix = train_dmatrix.slice(subsample)

        return new_train_dmatrix

    def subsample_indices(self, predictions: ndarray, train_dmatrix: xgb.DMatrix) -> ndarray:
        gradients, hessians = super().grad_and_hess(predictions, train_dmatrix)
        self.lambda_rate = self.mean_leaf_value() or self.mean_grad_value()
        regularized_gradients = np.sqrt(np.square(gradients) + self.lambda_rate * np.square(hessians))

        regularized_gradients = np.sqrt(np.square(gradients) + self.lambda_rate * np.square(hessians))

        subsample = np.argsort(regularized_gradients)[-int(len(regularized_gradients) * self.sample_rate):]

        return subsample

    def mean_leaf_value(self) -> Optional[float]:
        assert len(self.lastIterTree) == self.dimension
        num_leaves = 0
        sum_leaves = 0
        for tree in self.lastIterTree:
            sum = 0
            num = 0
            for line in tree.split("\n"):
                if "leaf" in line:
                    parts = line.split(",")
                    for part in parts:
                        if "leaf=" in part:
                            leaf_value = float(part.split('=')[1])
                            sum += np.square(leaf_value)
                            num += 1
            if num_leaves == 0:
                num_leaves = num  
            else: 
                assert num_leaves == num
            sum_leaves += np.sqrt(sum)
        
        return sum_leaves / num_leaves if num_leaves != 0 and sum_leaves != 0 else None
        
    def mean_grad_value(self, gradients: np.ndarray) -> float:
        return np.sum(np.sqrt(np.square(gradients))) / len(gradients)


def mvs_adaptive(lambda_rate = 0.1, sample_rate = 0.1):
    dataset = WineQualityDataloader(IidPartitioner(3))
    train_dmatrix, _ = dataset.get_train_dmatrix()
    test_dmatrix, _ = dataset.get_test_dmatrix(None)
    
    bst = xgb.Booster(params, [train_dmatrix])
    results_list = []
    for i in range(10):
        preds = bst.predict(train_dmatrix, output_margin=True, training=True)

        gradients, hessians = rmse_obj(preds, train_dmatrix)
        trees = bst.get_dump(dump_format="text")

        if len(trees) > 0:
            mean_leaf = mean_leaf_value([trees[-1]], 1)
            mean_grad = mean_grad_value(gradients)
            print("mean_leaf:", mean_leaf, ", mean_grad: ", mean_grad)
            lambda_rate = mean_leaf or mean_grad
        else:
            mean_grad = mean_grad_value(gradients)
            print("mean_grad: ", mean_grad)
            lambda_rate = mean_grad
        
        print("lambda_rate: ", lambda_rate)

        regularized_gradients = np.sqrt(np.square(gradients) + lambda_rate * np.square(hessians))

        indices = np.argsort(regularized_gradients)
        print(regularized_gradients[indices[-1]])

        subsample = indices[-int(len(regularized_gradients) * sample_rate):]

        new_train_dmatrix = train_dmatrix.slice(subsample)

        bst.update(new_train_dmatrix, i)
        evaluate = bst.eval_set([(test_dmatrix, "test")])
        print(evaluate)
        results_list.append(float(evaluate.split(':')[1]))
        
        plot_labels(3, WineQualityDataloader(IidPartitioner(3)), MVS(rmse_obj), bst, i)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_list, linewidth=2, label='wine_quality_adaptive')

    fontsize = 18
    plt.legend(fontsize=fontsize, loc='lower right')
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Rounds', fontsize=fontsize)
    plt.ylabel('AUC', fontsize=fontsize)

    plt.savefig('wine_quality_adaptive.png')

def mean_leaf_value(lastIterTree, dimension) -> Optional[float]:
    num_leaves = 0
    sum_leaves = 0
    for tree in lastIterTree:
        sum = 0
        num = 0
        for line in tree.split("\n"):
            if "leaf" in line:
                parts = line.split(",")
                for part in parts:
                    if "leaf=" in part:
                        print(part)
                        leaf_value = float(part.split('=')[1])
                        sum += np.square(leaf_value)
                        num += 1
        if num_leaves == 0:
            num_leaves = num  
        else: 
            assert num_leaves == num
        sum_leaves += np.sqrt(sum)
    
    return sum_leaves / num_leaves if num_leaves != 0 and sum_leaves != 0 else None
    
def mean_grad_value(gradients: np.ndarray) -> float:
    return np.sum(np.sqrt(np.square(gradients))) / len(gradients)

# mvs_adaptive()