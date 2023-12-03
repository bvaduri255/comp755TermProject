import numpy as np 
import pandas as pd

def make_splits(dataset_path, num_splits, shuffle=True):
    # Load dataset into matrix, shuffle, split features and labels
    full_data = np.genfromtxt(dataset_path, delimiter=',')[1:, :] # Remove first row of headerscl

    #full_data = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.], [5., 5., 5.]])
    if shuffle:
        np.random.shuffle(full_data)

    # Divide dataset into splits. Last split will get num_splits + (N % num_splits)
    N = len(full_data)
    split_size = N // num_splits
    last_split_size = split_size + N % num_splits
    splits = []

    for i in range(num_splits-1):
        start_idx, end_idx = i, i+split_size
        current_split = full_data[start_idx:end_idx, :]
        splits.append(current_split)
    splits.append(full_data[-last_split_size:, :])

    return splits


def write_splits(dataset_name, train_eval_test_percentages, splits):
    for i in range(len(splits)):
        split_instances = splits[i]

        # Find number of instances in train, eval, and test splits
        N = len(split_instances)
        assert np.sum(train_eval_test_percentages) == 1
        train_p, eval_p, test_p = train_eval_test_percentages
        train_instances = int(np.floor(N*train_p))
        eval_instances = int(np.floor(N*eval_p))
        test_instances = N - train_instances - eval_instances

        # Write instances to file
        train_output_path = f"{dataset_name}/split{i}_train"
        train_split = split_instances[:train_instances, :]
        pd.DataFrame(train_split).to_csv(train_output_path, index=False, header=False)

        eval_output_path = f"{dataset_name}/split{i}_eval"
        eval_split = split_instances[train_instances:train_instances+eval_instances, :]
        pd.DataFrame(eval_split).to_csv(eval_output_path, index=False, header=False)

        test_output_path = f"{dataset_name}/split{i}_test"
        test_split = split_instances[:-(train_instances+eval_instances), :]
        pd.DataFrame(test_split).to_csv(test_output_path, index=False, header=False)
    

if __name__ == "__main__":
    # Name of dataset folder 
    DATASET_NAME = "heart_data"

    # Relative path of the data file
    DATASET_PATH = "heart.csv"

    # Number of splits
    NUM_SPLITS = 3

    # Train-eval-test split
    TRAIN_SPLIT = 0.7
    VALID_SPLIT = 0.1
    TEST_SPLIT = 0.2

    TRAIN_EVAL_TEST_PERCENTAGES = [0.7, 0.1, 0.2]
    splits = make_splits(DATASET_PATH, NUM_SPLITS, shuffle=False)
    write_splits(DATASET_NAME, TRAIN_EVAL_TEST_PERCENTAGES, splits)
