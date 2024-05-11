from dataloader.multiclass_dataloader import CovertypeDataloader
from dataloader.binary_dataloader import HiggsDataloader
from flwr_datasets.partitioner import ExponentialPartitioner
from subsampling.mvs import MVS
import xgboost as xgb

import matplotlib.pyplot as plt


def mvs_simulation():
    num_clients = 5
    dataset = CovertypeDataloader(ExponentialPartitioner(num_clients))
    mvs = MVS(dataset.get_objective())

    for client in range(num_clients):
        train_dmatrix, _ = dataset.get_train_dmatrix(client)
        test_dmatrix, _ = dataset.get_test_dmatrix(None)
        bst = xgb.Booster(dataset.get_params(), [train_dmatrix])

        preds = bst.predict(train_dmatrix, output_margin=True, training=True)
        new_train_dmatrix = mvs.subsample(preds, train_dmatrix)
        bst.update(new_train_dmatrix, bst.num_boosted_rounds())

        bst.save_model(f"_static/model_{bst.num_boosted_rounds()}_{client}.json")

        evaluate = bst.eval_set([(test_dmatrix, "test")])
        print("Validation: ", evaluate)


def mvs_simulation_centralized():
    num_clients = 5
    dataset = HiggsDataloader(ExponentialPartitioner(num_clients))
    mvs = MVS(dataset.get_objective(), sample_rate=0.5)
    train_dmatrix, _ = dataset.get_train_dmatrix()
    test_dmatrix, _ = dataset.get_test_dmatrix(None)
    bst = xgb.Booster(dataset.get_params(), [train_dmatrix])

    for i in range(10):
        preds = bst.predict(train_dmatrix, output_margin=True, training=True)
        new_train_dmatrix = mvs.subsample(preds, train_dmatrix)
        bst.update(new_train_dmatrix, bst.num_boosted_rounds())

        evaluate = bst.eval_set([(test_dmatrix, "test")])
        print("Validation: ", evaluate)

def visualise():
    data = {
        "0.1": [0.8837, 0.7615, 0.7742, 0.7769, 0.7822, 0.8059, 0.8355, 0.8209, 0.8422, 0.8388],
        "0.2": [0.8905, 0.7690, 0.7800, 0.7895, 0.8496, 0.8511, 0.8515, 0.8731, 0.8565, 0.8714],
        "0.3": [0.8881, 0.7954, 0.8092, 0.8195, 0.8528, 0.8650, 0.8754, 0.8851, 0.8958, 0.9059],
        "0.4": [0.8871, 0.8135, 0.8199, 0.8438, 0.8828, 0.8945, 0.8956, 0.9050, 0.9123, 0.9168],
        "0.5": [0.8903, 0.8849, 0.8918, 0.8988, 0.9048, 0.9116, 0.9136, 0.9159, 0.9177, 0.9188],
        "0.6": [0.8915, 0.9002, 0.9022, 0.9030, 0.9113, 0.9140, 0.9170, 0.9176, 0.9182, 0.9187],
        "0.7": [0.8900, 0.8996, 0.9011, 0.9017, 0.9097, 0.9124, 0.9134, 0.9148, 0.9158, 0.9159],
        "0.8": [0.8964, 0.8989, 0.9024, 0.9059, 0.9064, 0.9068, 0.9061, 0.9051, 0.9045, 0.9036],
        "0.9": [0.8983, 0.8952, 0.8946, 0.8930, 0.8915, 0.8906, 0.8902, 0.8892, 0.8887, 0.8879],
        "1.0": [0.8975, 0.8968, 0.8945, 0.8926, 0.8905, 0.8891, 0.8881, 0.8877, 0.8873, 0.8870]
    }
    plt.figure(figsize=(10, 6))
    for key, values in data.items():
        plt.plot(values, linewidth=2, label=key)

    fontsize = 18
    plt.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=(1, 0))
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('FL 3 clients exponential bagging covertype', fontsize=20)
    plt.xlabel('Rounds', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('_static/xgboost_bagging_covertype.png')


visualise()
