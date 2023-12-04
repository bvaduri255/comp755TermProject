import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def make_splits_by_clustering(dataset_path, num_splits):
    # Load dataset and check for headers
    try:
        full_data = pd.read_csv(dataset_path, header=0).to_numpy()
    except pd.errors.EmptyDataError:
        full_data = np.genfromtxt(dataset_path, delimiter=',')[1:, :]  # Assuming first row is header

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_splits, random_state=42).fit(full_data)
    labels = kmeans.labels_

    # Create clusters
    clusters = [full_data[labels == i] for i in range(num_splits)]
    return clusters

def write_splits(dataset_name, train_eval_test_percentages, clusters):
    for i, cluster in enumerate(clusters):
        # Find number of instances in train, eval, and test splits
        N = len(cluster)
        assert np.sum(train_eval_test_percentages) == 1
        train_p, eval_p, test_p = train_eval_test_percentages
        train_instances = int(np.floor(N * train_p))
        eval_instances = int(np.floor(N * eval_p))

        # Write instances to file
        train_output_path = f"{dataset_name}/split{i}_train.csv"
        pd.DataFrame(cluster[:train_instances, :]).to_csv(train_output_path, index=False, header=False)

        eval_output_path = f"{dataset_name}/split{i}_eval.csv"
        eval_split = cluster[train_instances:train_instances + eval_instances, :]
        pd.DataFrame(eval_split).to_csv(eval_output_path, index=False, header=False)

        test_output_path = f"{dataset_name}/split{i}_test.csv"
        test_split = cluster[train_instances + eval_instances:, :]
        pd.DataFrame(test_split).to_csv(test_output_path, index=False, header=False)

if __name__ == "__main__":
    DATASET_NAME = "heart_data"
    DATASET_PATH = "heart.csv"
    NUM_SPLITS = 3
    TRAIN_EVAL_TEST_PERCENTAGES = [0.7, 0.1, 0.2]

    clusters = make_splits_by_clustering(DATASET_PATH, NUM_SPLITS)
    write_splits(DATASET_NAME, TRAIN_EVAL_TEST_PERCENTAGES, clusters)
