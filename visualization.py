import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from scipy.spatial import ConvexHull
from dataloader import Dataloader
from subsampling_strategy import SubsamplingStrategy

def plot_labels_(num_clients, dataloader: Dataloader, subsampling_strategy: SubsamplingStrategy, model: xgb.Booster, round: int):
    sampling_values = [0.2, 0.1, 0.05]
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
            class_counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
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
    plt.savefig(f"_static/sampling_values_{round}.png")
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
    important_features = model.get_score(importance_type='gain')
    print(important_features)
    xgb.plot_tree(model, rankdir='LR')
    plt.savefig("_static/tree_plot.png")


def plot__labels(num_clients, dataloader: Dataloader, subsampling_strategy: SubsamplingStrategy, model: xgb.Booster, round: int):
    markers = ['o', 's', 'x'] 
    colors = ['red', 'green', 'blue']
    xgb.plot_tree(model, rankdir='LR')
    plt.savefig("_static/tree_plot.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    important_features = model.get_score(importance_type='gain')
    print("Important features: ", important_features)

    sorted_features = sorted(important_features.items(), key=lambda x: x[1], reverse=True)

    highest_2_features = sorted_features[:2]
    x, y = tuple(int(key.replace('f', '')) for key, value in highest_2_features)
    centralized_dataset, _ = dataloader.get_train_dmatrix(None)
    centralized_labels = centralized_dataset.get_label()
    centralized_x = centralized_dataset.get_data()[:, x].toarray()
    centralized_y = centralized_dataset.get_data()[:, y].toarray()

    for label in range(3):
        class_indices = centralized_labels == label
        axs[0].scatter(
            centralized_x[class_indices],
            centralized_y[class_indices],
            marker=markers[label],
            color=colors[label],
        )
        axs[1].scatter(
            centralized_x[class_indices],
            centralized_y[class_indices],
            marker=markers[label],
            color=colors[label],
        )

    centralized_predictions = model.predict(centralized_dataset, output_margin=True, training=True)
    centralized_subsample_indices = subsampling_strategy.subsample_indices(centralized_predictions, centralized_dataset)
    centralized_x_subsample = centralized_x[centralized_subsample_indices]
    centralized_y_subsample = centralized_y[centralized_subsample_indices]

    points = np.column_stack((centralized_x_subsample, centralized_y_subsample))
    hull = ConvexHull(points)
    hull_vertices = hull.vertices.tolist() + [hull.vertices[0]]

    axs[0].plot(
        points[hull_vertices, 0], points[hull_vertices, 1],
        linestyle='--',
        color="red",
    )
    
    axs[0].set_title('Centralized Learning')
    axs[1].set_title('Federated Learning')

    for client in range(num_clients):
        partition, _ = dataloader.get_train_dmatrix(client)
        predictions = model.predict(partition, output_margin=True, training=True)
        dmatrix = subsampling_strategy.subsample(predictions, partition)
        fed_x = dmatrix.get_data()[:, x].toarray()
        fed_y = dmatrix.get_data()[:, y].toarray()
        points = np.column_stack((fed_x, fed_y))
        print(points.shape)
        print("here")
        hull = ConvexHull(points)
        hull_vertices = hull.vertices.tolist() + [hull.vertices[0]]

        axs[1].plot(
            points[hull_vertices, 0], points[hull_vertices, 1],
            linestyle='--',
            color=f'C{client}',
            label=f'Client {client}'
        )

    plt.tight_layout()
    plt.savefig(f"_static/sampling_values_{round}.png")
    
    print(">>> Sampling plot created")


def plot_labels(num_clients, dataloader: Dataloader, subsampling_strategy: SubsamplingStrategy, model: xgb.Booster, round: int):
    markers = ['o', 's', 'x'] 
    colors = ['red', 'green', 'blue']

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    important_features = model.get_score(importance_type='gain')

    sorted_features = sorted(important_features.items(), key=lambda x: x[1], reverse=True)

    highest_2_features = sorted_features[:2]
    x, y = tuple(int(key.replace('f', '')) for key, value in highest_2_features)
    centralized_dataset, _ = dataloader.get_train_dmatrix(None)
    centralized_x = centralized_dataset.get_data()[:, x].toarray()
    centralized_y = centralized_dataset.get_data()[:, y].toarray()

    centralized_predictions = model.predict(centralized_dataset, output_margin=True, training=True)
    centralized_subsample_indices = subsampling_strategy.subsample_indices(centralized_predictions, centralized_dataset)
    centralized_x_subsample = centralized_x[centralized_subsample_indices]
    centralized_y_subsample = centralized_y[centralized_subsample_indices]

    axs[0].scatter(
        centralized_x_subsample,
        centralized_y_subsample,
        marker='o',
        color='red',
    )
    
    axs[0].set_title('Centralized Learning')
    axs[1].set_title('Federated Learning')

    for client in range(num_clients):
        partition, _ = dataloader.get_train_dmatrix(client)
        predictions = model.predict(partition, output_margin=True, training=True)
        dmatrix = subsampling_strategy.subsample(predictions, partition)
        fed_x = dmatrix.get_data()[:, x].toarray()
        fed_y = dmatrix.get_data()[:, y].toarray()

        intersection_mask = np.isin(fed_x, centralized_x_subsample)
        intersection_mask_y = np.isin(fed_y, centralized_y_subsample)
        
        only_mask = ~ intersection_mask

        intersection_x = fed_x[intersection_mask]
        intersection_y = fed_y[intersection_mask]

        only_x = fed_x[only_mask]
        only_y = fed_y[only_mask]

        axs[1].scatter(
            intersection_x,
            intersection_y,
            marker='x',
            color='black',
        )

        axs[1].scatter(
            only_x,
            only_y,
            marker='o',
            color='red',
        )

    plt.tight_layout()
    plt.savefig(f"_static/sampling_values_{round}.png")
    
    print(">>> Sampling plot created")


def plot_labels___(num_clients, dataloader: Dataloader, subsampling_strategy: SubsamplingStrategy, model: xgb.Booster, round: int):
    markers = ['o', 's', 'x'] 
    colors = ['red', 'green', 'blue']

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    centralized_dataset, _ = dataloader.get_train_dmatrix(None)
    centralized_predictions = model.predict(centralized_dataset, output_margin=True, training=True)
    print(centralized_predictions[0])
    centralized_labels = centralized_dataset.get_label()
    print(centralized_labels)
    centralized_x, centralized_y = subsampling_strategy.grad_and_hess(centralized_predictions, centralized_dataset)
    centralized_subsample_indices = subsampling_strategy.subsample_indices(centralized_predictions, centralized_dataset)

    for label in range(3):
        class_indices = centralized_labels == label
        axs[0].scatter(
            centralized_x[class_indices],
            centralized_y[class_indices],
            marker=markers[label],
            color=colors[label],
        )
        axs[1].scatter(
            centralized_x[class_indices],
            centralized_y[class_indices],
            marker=markers[label],
            color=colors[label],
        )
    
    centralized_x_subsample_raw = centralized_x[centralized_subsample_indices]
    centralized_y_subsample_raw = centralized_y[centralized_subsample_indices]
    print(centralized_x_subsample_raw)
    centralized_x_subsample = np.abs(centralized_x_subsample_raw[centralized_x_subsample_raw < 0])
    centralized_y_subsample = np.abs(centralized_y_subsample_raw[centralized_x_subsample_raw < 0])
    print(centralized_x_subsample)
    print(centralized_y_subsample)
    #points = np.column_stack((centralized_x_subsample, centralized_y_subsample))
    #print(points.shape)
    #print("here")
    #hull = ConvexHull(points)
    #hull_vertices = hull.vertices.tolist() + [hull.vertices[0]]

    #axs[0].plot(
    #    points[hull_vertices, 0], points[hull_vertices, 1],
    #    linestyle='--',
    #    color=f'C{client}',
    #    label=f'Client {client}'
    #)
    rectangle = plt.Rectangle((centralized_x_subsample.min(), centralized_y_subsample.min()), centralized_x_subsample.max() - centralized_x_subsample.min(), centralized_y_subsample.max()-centralized_y_subsample.min(), color='red', fill=False, linestyle='--')
    axs[0].add_artist(rectangle)
    axs[0].set_title('Centralized Learning')
    axs[1].set_title('Federated Learning')

    for client in range(num_clients):
        partition, _ = dataloader.get_train_dmatrix(client)
        predictions = model.predict(partition, output_margin=True, training=True)
        subsample_indices = subsampling_strategy.subsample_indices(predictions, partition)
        subsample = partition.slice(subsample_indices)
        fed_x_raw, fed_y_raw = subsampling_strategy.grad_and_hess(predictions[subsample_indices], subsample)
        fed_x = np.abs(fed_x_raw[fed_x_raw < 0])
        fed_y = np.abs(fed_y_raw[fed_x_raw < 0])
        rectangle = plt.Rectangle((fed_x.min(), fed_y.min()), fed_x.max() - fed_x.min(), fed_y.max()-fed_y.min(), color='red', fill=False, linestyle='--')
        axs[1].add_artist(rectangle)
        #points = np.column_stack((fed_x, fed_y))
        #hull = ConvexHull(points)
        #hull_vertices = hull.vertices.tolist() + [hull.vertices[0]]

        #axs[1].plot(
        #    points[hull_vertices, 0], points[hull_vertices, 1],
        #    linestyle='--',
        #    color=f'C{client}',
        #    label=f'Client {client}'
        #)

    plt.tight_layout()
    plt.savefig(f"_static/sampling_values_{round}.png")
    
    print(">>> Sampling plot created")