import numpy as np
import xgboost as xgb
from thesis_dataset import create_centralized_dataset, ThesisDataset
from goss import params, softprob_obj

def minimal_variance_sampling(lambda_rate, sample_rate):
    train_dmatrix, test_dmatrix = create_centralized_dataset(ThesisDataset.IRIS.value)
    
    bst = xgb.Booster(params, [train_dmatrix])

    preds = bst.predict(train_dmatrix, output_margin=True, training=True)
    print(preds.shape)

    gradients, hessians = softprob_obj(preds, train_dmatrix)

    regularized_gradients = np.sqrt(np.square(gradients) + lambda_rate * np.square(hessians))