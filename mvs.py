import numpy as np
import xgboost as xgb
from thesis_dataset import create_centralized_dataset, ThesisDataset
from utils import params, softprob_obj, rmse_obj
from subsampling_strategy import SubsamplingStrategy
from typing import Optional
from flwr_datasets.partitioner import IidPartitioner
from wine_quality_dataloader import WineQualityDataloader
from visualization import plot_tree, plot_labels

def calculate_threshold(candidates, sum_small, num_large, sample_size):
    threshold = candidates[0]
    middle_begin = [x for x in candidates if x < threshold]
    middle_end = [x for x in candidates if x <= threshold]
    large_array = [x for x in candidates if x > threshold]

    sum_small_update = np.sum(middle_begin)
    num_large_update = len(large_array)
    num_middle = len(middle_end) - len(middle_begin)
    sum_middle = threshold * num_middle

    estimated_sample_size = (sum_small + sum_small_update) / threshold + num_large + num_large_update + num_middle
    print("estimated_sample_size: (", sum_small, " + ", sum_small_update, ") / ", threshold, " + ", num_large, " + ", num_large_update, " + ", num_middle, " = ", estimated_sample_size)
    if estimated_sample_size > sample_size:
        if len(large_array) > 0:
            print("estimated > sample_size, large_array size > 0: ", sum_small, " += ", sum_middle, " + ", sum_small_update)
            sum_small += sum_middle + sum_small_update
            return calculate_threshold(large_array, sum_small, num_large, sample_size)
        else:
            print("estimated > sample_size, large_array size = 0: (", sum_small, " + ", sum_small_update, " + ", sum_middle, ") / (", sample_size, " - ", num_large, ")")
            return (sum_small + sum_small_update + sum_middle) / (sample_size * 10 - num_large)
    else:
        if len(middle_begin) > 0:
            print("estimated <= sample_size, small_array size > 0: ", num_large, " += ", num_large_update, " + ", num_middle)
            num_large += num_large_update + num_middle
            return calculate_threshold(middle_begin, sum_small, num_large, sample_size)
        else:
            print("estimated <= sample_size, small_array size = 0: ", sum_small / (sample_size - num_large - num_middle - num_large_update))
            return sum_small / (sample_size - num_large - num_middle - num_large_update)

class MVS(SubsamplingStrategy):

    def __init__(self, objective, lambda_rate = 0.1, sample_rate = 0.1) -> None:
        self.lambda_rate = lambda_rate
        self.sample_rate = sample_rate
        self.objective = objective

    def subsample(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> xgb.DMatrix:
        subsample = self.subsample_indices(predictions, train_dmatrix)

        new_train_dmatrix = train_dmatrix.slice(subsample)

        return new_train_dmatrix
    
    def subsample_indices(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> np.ndarray:
        gradients, hessians = self.grad_and_hess(predictions, train_dmatrix)
        regularized_gradients = np.sqrt(np.square(gradients) + self.lambda_rate * np.square(hessians))

        regularized_gradients = np.sqrt(np.square(gradients) + self.lambda_rate * np.square(hessians))

        subsample = np.argsort(regularized_gradients)[-int(len(regularized_gradients) * self.sample_rate):]

        return subsample

    def global_sampling(self, grad_hess_dict: dict[int, list[(float, float)]]) -> dict[int, list[int]]:
        sampling_values = {}
        all_gradients = []
        all_hessians = []

        for id, (grad, hess) in grad_hess_dict.items():
            all_gradients.extend(grad)
            all_hessians.extend(hess)

        all_gradients = np.array(all_gradients)
        all_hessians = np.array(all_hessians)

        mask = all_gradients < 0
        regularized_gradients = np.sqrt(np.square(all_gradients[mask]) + self.lambda_rate * np.square(all_hessians[mask]))

        num_samples = int(len(regularized_gradients) * self.sample_rate)

        subsample_indices = np.argsort(regularized_gradients)[-num_samples:]

        start_index = 0
        for id, (grad, hess) in grad_hess_dict.items():
            num_elements = len(grad)

            end_index = start_index + num_elements
            current_subsample_indices = [
                idx - start_index
                for idx in subsample_indices
                if start_index <= idx < end_index
            ]

            sampling_values[id] = current_subsample_indices

            start_index = end_index

        return sampling_values
    
    def grad_and_hess(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        return self.objective(predictions, train_dmatrix)
    

def minimal_variance_sampling(lambda_rate = 0.1, sample_rate = 0.1):
    dataset = WineQualityDataloader(IidPartitioner(3))
    train_dmatrix, _ = dataset.get_train_dmatrix()
    print(train_dmatrix.get_label())
    test_dmatrix, _ = dataset.get_test_dmatrix(None)
    print(test_dmatrix.get_label())
    
    bst = xgb.Booster(params, [train_dmatrix])
    for i in range(5):
        preds = bst.predict(train_dmatrix, output_margin=True, training=True)

        gradients, hessians = rmse_obj(preds, train_dmatrix)

        regularized_gradients = np.sqrt(np.square(gradients) + lambda_rate * np.square(hessians))

        subsample = np.argsort(regularized_gradients)[-int(len(regularized_gradients) * sample_rate):]

        new_train_dmatrix = train_dmatrix.slice(subsample)

        bst.update(new_train_dmatrix, i)
        print(bst.eval_set([(test_dmatrix, "test")]))

        plot_tree(bst)
        trees_dump = bst.get_dump(dump_format="text")
        print(trees_dump[-1])
        
        plot_labels(3, WineQualityDataloader(IidPartitioner(3)), MVS(rmse_obj), bst, i)

minimal_variance_sampling()
    