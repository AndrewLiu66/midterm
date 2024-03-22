import numpy as np
from scipy.stats import mode
from math import log2, ceil

def calculate_mec_multiclass(data, labels):
    d = data.shape[1]
    thresholds = 0

    # Create a table combining data and labels
    table = np.column_stack((data, labels))

    # Sort the table based on the data (not including the label column)
    sorted_table = table[table[:, :-1].sum(axis=1).argsort()]  # Sort by sum of all dimensions

    # Find the unique classes (for multi-class classification)
    unique_classes = np.unique(labels)
    class_transitions = {class_label: 0 for class_label in unique_classes}
    current_classes = sorted_table[0, -len(unique_classes):]


    for row in sorted_table:
        for i, class_label in enumerate(unique_classes):
            if row[-len(unique_classes) + i] != current_classes[i]:
                current_classes[i] = row[-len(unique_classes) + i]
                class_transitions[class_label] += 1

    thresholds = sum(class_transitions.values())

    min_thresholds = log2(thresholds + 1)

    mec = (min_thresholds * (d + len(unique_classes))) + (min_thresholds + 1)

    return mec

# Example data and labels for multi-class classification
example_data = np.array([[0.1, 0.2],
                         [0.4, 0.5],
                         [0.3, 0.3],
                         [0.6, 0.7]])
example_labels_multiclass = np.array([[0, 1],
                                      [1, 0],
                                      [0, 1],
                                      [1, 0]])

# Calculate the memory-equivalent capacity for the example multi-class data
mec_value_multiclass = calculate_mec_multiclass(example_data, example_labels_multiclass)
print(mec_value_multiclass)
