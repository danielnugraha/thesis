import numpy as np
import xgboost as xgb
from thesis_dataset import create_centralized_dataset, ThesisDataset
from utils import softprob_obj, params
from subsampling_strategy import SubsamplingStrategy


# start learning after 1.0 / learning rate

def gradient_based_one_side_sampling(dataset: str = "", a = 0.05, b = 0.05, ):
    train_dmatrix, test_dmatrix = create_centralized_dataset(ThesisDataset.IRIS.value)
    
    bst = xgb.Booster(params, [train_dmatrix])

    fact = (1 - a) / b
    topN = a * train_dmatrix.num_row()
    randN = b * train_dmatrix.num_row()
    
    preds = bst.predict(train_dmatrix, output_margin=True, training=True)

    gradients, _ = softprob_obj(preds, train_dmatrix)

    weights = np.ones_like(train_dmatrix.get_label())
    sorted_indices = np.argsort(np.abs(np.sum(gradients, axis=1, keepdims=False)))
    topSet = sorted_indices[:int(topN)]
    randSet = np.random.choice(sorted_indices[int(topN):], int(randN), replace=False)
    usedSet = np.concatenate([topSet, randSet])
    weights[randSet] *= fact
    new_train_dmatrix = train_dmatrix.slice(usedSet)
    new_train_dmatrix.set_weight(weights[usedSet])

    bst.update(new_train_dmatrix, 1)
    print(bst.eval_set([(test_dmatrix, "test")]))
    return bst

gradient_based_one_side_sampling()

class GOSS(SubsamplingStrategy):

    def __init__(self, train_dmatrix, lambda_rate, sample_rate) -> None:
        self.train_dmatrix = train_dmatrix
        self.lambda_rate = lambda_rate
        self.sample_rate = sample_rate
        self.dataset = None

    def subsample(self, bst):
        return super().subsample(bst)
    
    @property
    def dataset(self):
        return self.dataset