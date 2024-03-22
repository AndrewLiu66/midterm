import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def experiment_with_dimensionality_multi_class(D, n_classes):
    n_functions = 2 ** D
    memorization_points = []

    for _ in range(n_functions):
        X, y = make_classification(n_samples=1000, n_features=D, n_redundant=0, n_informative=D, n_clusters_per_class=1, n_classes=n_classes, random_state=None)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        if accuracy == 1.0:
            memorization_points.append(len(X_train))

    avg_mem_size = np.mean(memorization_points)
    print(f"d={D}, classes={n_classes}: n_full={2**D}, Avg. req. points for memorization n_avg={avg_mem_size:.2f}, n_full/n_avg={(2**D)/avg_mem_size}")

for D in [2, 4, 8]:
    experiment_with_dimensionality_multi_class(D, 3)  # Example with 3 classes