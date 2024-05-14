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
    dataset = CovertypeDataloader(ExponentialPartitioner(num_clients))
    mvs = MVS(dataset.get_objective(), sample_rate=1.0)
    train_dmatrix, _ = dataset.get_train_dmatrix()
    test_dmatrix, _ = dataset.get_test_dmatrix(None)
    bst = xgb.Booster(dataset.get_params(), [train_dmatrix])

    for i in range(30):
        preds = bst.predict(train_dmatrix, output_margin=True, training=True)
        new_train_dmatrix = mvs.subsample(preds, train_dmatrix)
        bst.update(new_train_dmatrix, bst.num_boosted_rounds())

        evaluate = bst.eval_set([(test_dmatrix, "test")])
        print("Validation: ", evaluate)

def visualise():
    data = {
        "0.1": [0.7781, 0.7947, 0.8013, 0.8047, 0.8063, 0.8076, 0.8084, 0.8089, 0.8092, 0.8097, 0.8097, 0.8096, 0.8098, 0.8096, 0.8097, 0.8099, 0.81, 0.8098, 0.8096, 0.8097, 0.8093, 0.8091, 0.809, 0.8087, 0.8083, 0.808, 0.808, 0.8076, 0.8071, 0.8069],
        "0.2": [0.7781, 0.7949, 0.8015, 0.8056, 0.8083, 0.8098, 0.8109, 0.812, 0.8127, 0.8133, 0.8138, 0.814, 0.8144, 0.8148, 0.815, 0.8151, 0.8154, 0.8155, 0.8157, 0.8157, 0.8157, 0.8156, 0.8155, 0.8156, 0.8154, 0.8153, 0.8153, 0.8154, 0.8153, 0.8153],
        "0.3": [0.7784, 0.796, 0.803, 0.8072, 0.81, 0.8118, 0.8132, 0.8146, 0.8157, 0.8163, 0.817, 0.8175, 0.818, 0.8185, 0.8187, 0.819, 0.8192, 0.8194, 0.8197, 0.8201, 0.8202, 0.8203, 0.8205, 0.8205, 0.8205, 0.8204, 0.8205, 0.8203, 0.8203, 0.8203],
        "0.4": [0.7776, 0.7952, 0.8023, 0.8061, 0.8096, 0.8118, 0.8133, 0.8148, 0.8161, 0.8175, 0.8181, 0.8187, 0.8197, 0.82, 0.8205, 0.8211, 0.8213, 0.8216, 0.8218, 0.822, 0.8221, 0.8223, 0.8225, 0.8225, 0.8225, 0.8225, 0.8226, 0.8227, 0.8228, 0.8227],
        "0.5": [0.7775, 0.7935, 0.8018, 0.8064, 0.8091, 0.8116, 0.8136, 0.8151, 0.8164, 0.8176, 0.8184, 0.8191, 0.8199, 0.8204, 0.8209, 0.8212, 0.8215, 0.8218, 0.8222, 0.8225, 0.8227, 0.8228, 0.8229, 0.823, 0.8232, 0.8234, 0.8234, 0.8237, 0.8238, 0.8239],
        "0.6": [0.7766, 0.7938, 0.8013, 0.8075, 0.8106, 0.8123, 0.8138, 0.8153, 0.8166, 0.8178, 0.8188, 0.8198, 0.8205, 0.821, 0.8217, 0.8221, 0.8223, 0.8229, 0.8233, 0.8235, 0.8237, 0.824, 0.8242, 0.8243, 0.8245, 0.8249, 0.8249, 0.825, 0.825, 0.825],
        "0.7": [0.7763, 0.7949, 0.8017, 0.8066, 0.8087, 0.8118, 0.8139, 0.8155, 0.8167, 0.8179, 0.8188, 0.8197, 0.8204, 0.8211, 0.8219, 0.8224, 0.8227, 0.8231, 0.8236, 0.8241, 0.8244, 0.8247, 0.8248, 0.825, 0.8252, 0.8256, 0.8257, 0.8259, 0.826, 0.8261],
        "0.8": [0.7756, 0.7943, 0.8018, 0.8069, 0.8091, 0.8119, 0.8142, 0.8154, 0.8166, 0.8175, 0.8189, 0.8199, 0.8207, 0.8213, 0.8221, 0.8226, 0.8232, 0.8238, 0.8242, 0.8246, 0.8248, 0.8251, 0.8253, 0.8254, 0.8256, 0.8258, 0.8259, 0.826, 0.8264, 0.8264],
        "0.9": [0.7751, 0.7934, 0.801, 0.8055, 0.8083, 0.8107, 0.8132, 0.8148, 0.8158, 0.8173, 0.8185, 0.8191, 0.8197, 0.8208, 0.8216, 0.8221, 0.8226, 0.823, 0.8236, 0.8239, 0.8242, 0.8246, 0.8249, 0.8252, 0.8255, 0.8257, 0.8258, 0.826, 0.8263, 0.8263],
        "1.0": [0.7755, 0.7939, 0.801, 0.8056, 0.8081, 0.8118, 0.8138, 0.8152, 0.8166, 0.818, 0.8193, 0.8199, 0.8207, 0.8213, 0.8218, 0.8224, 0.8228, 0.8234, 0.8241, 0.8244, 0.8247, 0.8249, 0.8252, 0.8255, 0.8255, 0.8257, 0.826, 0.8261, 0.8263, 0.8264],
    }

    plt.figure(figsize=(12, 8))
    for key, values in data.items():
        plt.plot(values, linewidth=2, label=key)

    fontsize = 18
    plt.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=(1, 0))
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('FL 10 clients uniform bagging higgs', fontsize=20)
    plt.xlabel('Rounds', fontsize=fontsize)
    plt.ylabel('AUC', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('_static/xgboost_uniform_higgs_10.png')


mvs_simulation_centralized()
