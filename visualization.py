import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from dataloader import Dataloader
from subsampling_strategy import SubsamplingStrategy

def plot_labels(num_clients, dataloader: Dataloader, subsampling_strategy: SubsamplingStrategy, model: xgb.Booster):
    sampling_values = [0.85, 0.5, 0.3, 0.2, 0.1, 0.05]
    num_classes = dataloader.get_num_classes()
    color = plt.get_cmap("RdYlGn")(np.linspace(0.15, 0.85, num_classes))
    num_plots = len(sampling_values)
    fig, axs = plt.subplots(1, num_plots, figsize=(15, 5), sharey=True)

    pos = axs[0].get_position()
    pos.x0 += 0.1
    axs[0].set_position(pos)

    for i, sampling in enumerate(sampling_values):
        for client in range(num_clients):
            dataset, _ = dataloader.get_train_dmatrix(client)
            predictions = model.predict(dataset, output_margin=True, training=True)
            dmatrix = subsampling_strategy.subsample(predictions, dataset)
            labels = dmatrix.get_label()
            lefts = [0]
            axis = axs[i]
            class_counts = np.bincount(labels, minlength=num_classes)
            np.sum(class_counts > 0)

            class_distribution = class_counts.astype(np.float16) / len(labels)

            for idx, val in enumerate(class_distribution[:-1]):
                lefts.append(lefts[idx] + val)

            axis.barh(client, class_distribution, left=lefts, color=color)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel("Class distribution")
            axis.set_title(f"Sampling = {sampling}")

    fig.text(0, 0.5, "Client", va="center", rotation="vertical")
    plt.tight_layout()
    plt.savefig("../_static/sampling_values.png")
    print(">>> Sampling plot created")
    

def plot_feature_importance(model: xgb.Booster):
    important_features = model.get_score(importance_type='gain')

    sorted_features = sorted(important_features.items(), key=lambda x: x[1], reverse=True)

    top_40_features = sorted_features[:40]

    keys, values = zip(*top_40_features)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.barh(keys, values)

    ax.set_xlabel('Importance')
    ax.set_title('Top 40 Features by Importance')

    plt.savefig("../_static/important_features.png")


def plot_tree(model:xgb.Booster):
    xgb.plot_tree(model, rankdir='LR')
    plt.savefig("../_static/tree_plot.png")
