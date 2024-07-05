from dataloader import Dataloader, CodrnaDataloader, CustomerChurnDataloader, AbaloneDataloader, FairJobDataloader, CovertypeDataloader, HelenaDataloader, AirlinesDataloader, CompasDataloader, DionisDataloader, HiggsDataloader, RoadSafetyDataloader, JannisDataloader, WineQualityDataloader, AllstateClaimsSeverityDataloader, HouseSalesDataloader, DiamondsDataloader, YearPredictionMsdDataloader
from flwr_datasets.partitioner import ExponentialPartitioner
from subsampling.mvs import MVS
import xgboost as xgb
import os
import csv
import numpy as np

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
    dataset = CompasDataloader(ExponentialPartitioner(num_clients))
    sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    train_dmatrix, _ = dataset.get_train_dmatrix()
    test_dmatrix, _ = dataset.get_test_dmatrix(None)
    
    all_results = {}

    for sample_rate in sample_rates:
        eval_results = []
        mvs = MVS(dataset.get_objective(), sample_rate=sample_rate)
        bst = xgb.Booster(dataset.get_params(), [train_dmatrix])
        for i in range(10):
            preds = bst.predict(train_dmatrix, output_margin=True, training=True)
            new_train_dmatrix = mvs.subsample(preds, train_dmatrix)
            bst.update(new_train_dmatrix, bst.num_boosted_rounds())

            evaluate = bst.eval_set([(test_dmatrix, "test")])
            auc = round(float(evaluate.split("\t")[1].split(":")[1]), 4)
            eval_results.append(auc)
            print("Validation: ", evaluate)
            # bst.save_model(f'xgboost_{sample_rate}_{i}.json')

        print(eval_results)
        all_results[sample_rate] = eval_results
    
    print(all_results)

def load_dataset_try():
    num_clients = 5
    dataset = HiggsDataloader(ExponentialPartitioner(num_clients))
    train_dmatrix, _ = dataset.get_train_dmatrix(0)
    print(train_dmatrix.get_data().get_shape()[1])
    test_dmatrix, _ = dataset.get_test_dmatrix(None)

def dataset_analysis():
    num_clients = 5
    dataset = HiggsDataloader(ExponentialPartitioner(num_clients))
    dataset.dataset_analysis()

def visualise():
    data = {
        0.1: [1.8442, 1.5249, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265, 1.7775, 1.5265],
        0.2: [1.6134, 1.3474, 1.5821, 1.3411, 1.5766, 1.3374, 1.5752, 1.3365, 1.5736, 1.3362, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363, 1.5717, 1.3363],
        0.3: [1.1914, 1.0503, 1.0794, 0.9922, 1.097, 1.0144, 1.1195, 1.0342, 1.144, 1.0434, 1.1532, 1.048, 1.1492, 1.0487, 1.1455, 1.038, 1.1432, 1.0432, 1.14, 1.0418, 1.1556, 1.0566, 1.1604, 1.0579, 1.1593, 1.0529, 1.1606, 1.0606, 1.1585, 1.0582, 1.1622, 1.0592, 1.1558, 1.0514, 1.1471, 1.0614, 1.1568, 1.0607, 1.1576, 1.0592, 1.1515, 1.0577, 1.1618, 1.0626, 1.1631, 1.0672, 1.1626, 1.0671, 1.1654, 1.0682],
        0.4: [1.1158, 0.8992, 0.8996, 0.857, 0.8832, 0.8251, 0.875, 0.8388, 0.8651, 0.8327, 0.8775, 0.856, 0.9044, 0.8619, 0.8981, 0.8597, 0.8994, 0.8764, 0.908, 0.8998, 0.9275, 0.9043, 0.9376, 0.9017, 0.9262, 0.8759, 0.9146, 0.8685, 0.9157, 0.8769, 0.9201, 0.8883, 0.9353, 0.9017, 0.9497, 0.9061, 0.9421, 0.9053, 0.9452, 0.9033, 0.9406, 0.9145, 0.9474, 0.9266, 0.9559, 0.9395, 0.9771, 0.9545, 0.977, 0.9498],
        0.5: [1.0567, 0.7755, 0.7679, 0.6978, 0.7137, 0.665, 0.697, 0.6704, 0.6804, 0.6558, 0.6617, 0.6308, 0.641, 0.6269, 0.6375, 0.6236, 0.613, 0.606, 0.6089, 0.6011, 0.6129, 0.601, 0.5968, 0.6017, 0.595, 0.5977, 0.5916, 0.5882, 0.597, 0.5882, 0.5975, 0.5941, 0.601, 0.5987, 0.5956, 0.5951, 0.5914, 0.5967, 0.5924, 0.5822, 0.594, 0.5926, 0.5933, 0.5955, 0.5891, 0.5901, 0.5822, 0.5717, 0.5771, 0.5673],
        0.6: [0.9578, 0.7121, 0.69, 0.6598, 0.6632, 0.6288, 0.6217, 0.6118, 0.6102, 0.6092, 0.6053, 0.5994, 0.5984, 0.5976, 0.6028, 0.5844, 0.5902, 0.5822, 0.5798, 0.5791, 0.5795, 0.5689, 0.5681, 0.5738, 0.5695, 0.5761, 0.5798, 0.5766, 0.5727, 0.576, 0.5718, 0.5756, 0.5746, 0.5724, 0.5749, 0.5831, 0.5847, 0.5922, 0.5859, 0.5903, 0.5853, 0.5801, 0.5772, 0.5824, 0.5831, 0.5804, 0.5812, 0.5817, 0.5756, 0.5666],
        0.7: [0.8735, 0.6884, 0.6574, 0.6198, 0.6175, 0.596, 0.5867, 0.5827, 0.5777, 0.578, 0.5709, 0.5664, 0.5649, 0.5571, 0.5618, 0.5636, 0.5578, 0.5486, 0.545, 0.5397, 0.5358, 0.5308, 0.5213, 0.5174, 0.5188, 0.5206, 0.5217, 0.5205, 0.5153, 0.5086, 0.5122, 0.5163, 0.513, 0.5055, 0.4997, 0.4971, 0.5015, 0.4994, 0.5014, 0.5073, 0.5116, 0.5081, 0.5045, 0.5024, 0.5019, 0.5086, 0.5108, 0.5091, 0.5077, 0.5104],
        0.8: [0.7613, 0.6538, 0.6249, 0.6149, 0.6143, 0.6001, 0.589, 0.5762, 0.5662, 0.5621, 0.5614, 0.5518, 0.5542, 0.5488, 0.5465, 0.5441, 0.5456, 0.5452, 0.5422, 0.5314, 0.5283, 0.5264, 0.5283, 0.5285, 0.5296, 0.5237, 0.5212, 0.5183, 0.5152, 0.5154, 0.5189, 0.5145, 0.5129, 0.5149, 0.5161, 0.5189, 0.5209, 0.5214, 0.5204, 0.5182, 0.5198, 0.5217, 0.5184, 0.5181, 0.5177, 0.5168, 0.5159, 0.5107, 0.5107, 0.5094],
        0.9: [0.7309, 0.6407, 0.6219, 0.605, 0.5977, 0.5951, 0.5894, 0.5799, 0.5796, 0.5816, 0.5738, 0.5744, 0.572, 0.5719, 0.5755, 0.5672, 0.5592, 0.5567, 0.5587, 0.5637, 0.5606, 0.5544, 0.5468, 0.5474, 0.546, 0.5455, 0.5422, 0.5361, 0.5332, 0.5318, 0.5326, 0.5322, 0.5352, 0.5364, 0.5362, 0.5363, 0.5337, 0.5337, 0.5298, 0.5321, 0.5344, 0.5327, 0.5325, 0.533, 0.5318, 0.5328, 0.5284, 0.5309, 0.528, 0.5264],
        1.0: [0.681, 0.6483, 0.6305, 0.618, 0.6079, 0.6001, 0.5933, 0.5902, 0.589, 0.5803, 0.5707, 0.5656, 0.5584, 0.5526, 0.5525, 0.5527, 0.5564, 0.5525, 0.5518, 0.5477, 0.5492, 0.5513, 0.5545, 0.5536, 0.5561, 0.5536, 0.5547, 0.5509, 0.5441, 0.5492, 0.5475, 0.5459, 0.5461, 0.5508, 0.5534, 0.5546, 0.5525, 0.5562, 0.5544, 0.554, 0.5553, 0.5477, 0.5444, 0.5425, 0.5448, 0.5468, 0.55, 0.5521, 0.5555, 0.5601],
    }

    plt.figure(figsize=(12, 8))
    for key, values in data.items():
        plt.plot(values[:50], linewidth=2, label=key)
        print(f"{key}: [{','.join(map(str, values[:50]))}],")

    fontsize = 18
    plt.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=(1, 0))
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('Federated MVS - Wine_quality - 10 clients uniform local evaluation - 50 rounds', fontsize=fontsize)
    plt.xlabel('Rounds', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig('_static/wine_quality_mvs_10_uniform_50.png')

load_dataset_try()

# visualise()

# mvs_simulation_centralized()

dataset_analysis()

def calculate_average(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        numbers = [int(row[0]) for row in reader]
        return sum(numbers) / len(numbers) if numbers else 0

averages = []

# Assuming the CSV files are named '1.csv', '2.csv', ..., '100.csv'
#for i in range(1, 101):
#    filename = f'./_static/{i}.csv'
#    if os.path.exists(filename):
#        avg = calculate_average(filename)
#        averages.append(round(avg, 1))
#        os.remove(filename)
#    else:
#        averages.append(None)  # Or handle the missing file case as needed

# Output the list of averages
#print(averages)
#print("average: ", sum(averages) / len(averages))
