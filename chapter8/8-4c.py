import numpy as np
from scipy.stats import mode
from math import log2, ceil

def calculate_mec_regression(data, labels):

    d = data.shape[1]
    thresholds = 0
    table = np.column_stack((data, labels))

    sorted_table = table[table[:, :-1].sum(axis=1).argsort()]
    label_std = np.std(labels)
    if label_std == 0:
        thresholds = 1
    else:
        sorted_labels = np.sort(labels)
        label_changes = np.diff(sorted_labels)
        threshold_for_change = label_std / 2
        thresholds = 1 + np.sum(label_changes > threshold_for_change)

    # Calculate the minimum number of thresholds
    min_thresholds = log2(thresholds + 1)

    # Calculate the memory-equivalent capacity (MEC) for regression
    mec = (min_thresholds * (d + 1)) + (min_thresholds + 1)

    return mec

# Example usage of the function for regression:
example_data_regression = np.array([[0.1, 0.2],
                                    [0.4, 0.5],
                                    [0.3, 0.3],
                                    [0.6, 0.7]])
example_labels_regression = np.array([0.2, 0.5, 0.3, 0.7])

mec_value_regression = calculate_mec_regression(example_data_regression, example_labels_regression)
print(mec_value_regression)
