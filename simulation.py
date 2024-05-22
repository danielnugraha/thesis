from dataloader.multiclass_dataloader import CovertypeDataloader, LetterRecognitionDataloader
from dataloader.binary_dataloader import HiggsDataloader, RoadSafetyDataloader, MiniBooNEDataloader, EyeMovementsDataloader, JannisDataloader, CaliforniaDataloader
from dataloader.regression_dataloader import SuperconductDataloader, CpuActDataloader, DelaysZurichTransportDataloader, AllstateClaimsSeverityDataloader, HouseSalesDataloader, DiamondsDataloader
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
    dataset = HouseSalesDataloader(ExponentialPartitioner(num_clients))
    mvs = MVS(dataset.get_objective(), sample_rate=0.1)
    train_dmatrix, _ = dataset.get_train_dmatrix()
    test_dmatrix, _ = dataset.get_test_dmatrix(None)
    bst = xgb.Booster(dataset.get_params(), [train_dmatrix])
    eval_results = []

    for i in range(10):
        preds = bst.predict(train_dmatrix, output_margin=True, training=True)
        new_train_dmatrix = mvs.subsample(preds, train_dmatrix)
        bst.update(new_train_dmatrix, bst.num_boosted_rounds())

        evaluate = bst.eval_set([(test_dmatrix, "test")])
        auc = round(float(evaluate.split("\t")[1].split(":")[1]), 4)
        eval_results.append(auc)
        print("Validation: ", evaluate)

    print(eval_results)

def load_dataset_try():
    num_clients = 5
    dataset = DiamondsDataloader(ExponentialPartitioner(num_clients))
    train_dmatrix, _ = dataset.get_train_dmatrix()
    test_dmatrix, _ = dataset.get_test_dmatrix(None)

def visualise():
    data = {
        "1.0": [0.7769, 0.7833, 0.7867, 0.7895, 0.7919, 0.7938, 0.7956, 0.7971, 0.7984, 0.7997, 0.8009, 0.802, 0.8031, 0.804, 0.8052, 0.8059, 0.8068, 0.8077, 0.8088, 0.8096, 0.8105, 0.8113, 0.8121, 0.8128, 0.8135, 0.8141, 0.8147, 0.8153, 0.816, 0.8164, 0.817, 0.8174, 0.8179, 0.8184, 0.8188, 0.8192, 0.8196, 0.8199, 0.8203, 0.8206, 0.821, 0.8214, 0.8216, 0.8218, 0.8221, 0.8225, 0.8228, 0.8231, 0.8233, 0.8236],
        "0.9": [0.7764, 0.7847, 0.7904, 0.7945, 0.7974, 0.7992, 0.8006, 0.8025, 0.8037, 0.8044, 0.805, 0.8058, 0.8067, 0.8072, 0.8081, 0.8085, 0.8091, 0.8096, 0.8101, 0.8107, 0.8114, 0.8119, 0.8125, 0.813, 0.8137, 0.8142, 0.8148, 0.8153, 0.8157, 0.8162, 0.8166, 0.817, 0.8174, 0.8178, 0.8182, 0.8185, 0.8188, 0.8191, 0.8194, 0.8198, 0.8201, 0.8203, 0.8206, 0.8209, 0.8212, 0.8215, 0.8219, 0.8221, 0.8224, 0.8227],
        "0.8": [0.7755, 0.7746, 0.7898, 0.7971, 0.8005, 0.8023, 0.8042, 0.8061, 0.8074, 0.8086, 0.8093, 0.8103, 0.8107, 0.8109, 0.8111, 0.8116, 0.8119, 0.8121, 0.8121, 0.8124, 0.8126, 0.8127, 0.813, 0.8133, 0.8135, 0.8136, 0.8138, 0.814, 0.8144, 0.8146, 0.8147, 0.8149, 0.8151, 0.8155, 0.8156, 0.8159, 0.816, 0.8166, 0.8168, 0.8175, 0.8179, 0.8183, 0.8187, 0.8189, 0.8193, 0.8194, 0.8197, 0.82, 0.8204, 0.8207],
        "0.7": [0.7739, 0.7632, 0.7606, 0.7595, 0.7782, 0.7863, 0.7928, 0.7968, 0.7999, 0.8022, 0.8039, 0.8053, 0.8061, 0.8072, 0.8078, 0.8084, 0.8087, 0.8091, 0.8097, 0.8099, 0.81, 0.8103, 0.8105, 0.8105, 0.8106, 0.8106, 0.8107, 0.8108, 0.811, 0.8112, 0.8113, 0.8114, 0.8116, 0.8116, 0.8116, 0.8116, 0.8117, 0.8118, 0.8119, 0.8122, 0.8123, 0.8124, 0.8126, 0.8129, 0.8129, 0.8131, 0.8132, 0.8134, 0.8135, 0.8138]
    }

    plt.figure(figsize=(12, 8))
    for key, values in data.items():
        plt.plot(values, linewidth=2, label=key)

    fontsize = 18
    plt.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=(1, 0))
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('Centralized higgs max_depth 10', fontsize=20)
    plt.xlabel('Rounds', fontsize=fontsize)
    plt.ylabel('AUC', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('_static/xgboost_centralized_higg_max_depth_10.png')

# load_dataset_try()
# visualise()

mvs_simulation_centralized()
