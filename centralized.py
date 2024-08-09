from utils import generic_args_parser, instantiate_partitioner, instantiate_dataloader, instantiate_sampling_method
import xgboost as xgb

args = generic_args_parser()
sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

num_rounds = args.num_rounds
sampling_method = args.sampling_method
partitioner_type = args.partitioner_type
dataloader_str = args.dataloader
num_partitions = args.num_partitions

partitioner = instantiate_partitioner(
    partitioner_type=partitioner_type, num_partitions=num_partitions
)

dataloader = instantiate_dataloader(dataloader_str, partitioner)

train_dmatrix, num_train, = dataloader.get_train_dmatrix()
valid_dmatrix, num_val = dataloader.get_test_dmatrix(None)

for sample_rate in sample_rates:
    subsampling_strategy = instantiate_sampling_method(sampling_method, dataloader.get_objective(), sample_rate)
    params = dataloader.get_params()
    if subsampling_strategy is None:
        params.update({"subsample": sample_rate})
    
    print(params)
    
    eval_results = []

    bst = xgb.Booster(params, [train_dmatrix])

    for i in range(num_rounds):
        if subsampling_strategy is None:
            bst.update(train_dmatrix, bst.num_boosted_rounds())
        else:
            preds = bst.predict(train_dmatrix, output_margin=True, training=True)
            new_train_dmatrix = subsampling_strategy.subsample(preds, train_dmatrix)
            bst.update(new_train_dmatrix, bst.num_boosted_rounds())

        evaluate = bst.eval_set([(valid_dmatrix, "test")])
        auc = round(float(evaluate.split("\t")[1].split(":")[1]), 4)
        eval_results.append(auc)
        print("Validation round ", i, ": ", evaluate)
        # bst.save_model(f'xgboost_{sample_rate}_{i}.json')

    with open(f'_static/{dataloader_str}_{sampling_method}_{partitioner_type}_{sample_rate}_1.txt', 'w') as file:
        file.write(','.join(map(str, eval_results)))
    
    print(f"Centralized training for {dataloader_str} with {sampling_method} and {sample_rate} completed.")
