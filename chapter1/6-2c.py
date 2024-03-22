# Dataset 1: Moderate complexity (10 features, 500 samples)
# Dataset 2: Increased complexity (20 features, 1000 samples)
# Dataset 3: High complexity (30 features, 1500 samples)

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_evaluate_on_random_data(n_features, n_samples, clf, strategy_description):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=int(n_features/2), n_redundant=0, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf.fit(X_train, y_train)
    rules = export_text(clf, feature_names=[f'feature_{i}' for i in range(n_features)])
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nDataset: {n_features} features, {n_samples} samples")
    print(f"Strategy: {strategy_description}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of Rules: {rules.count('class')}")
    print("-" * 50)

# Configuration parameters for the datasets
datasets = [
    (10, 500),
    (20, 1000),
    (30, 1500)
]

strategies = [
    (DecisionTreeClassifier(max_depth=3, random_state=42), "Limited Tree Depth (max_depth=3)"),
    (DecisionTreeClassifier(min_samples_leaf=5, random_state=42), "Increased Min Samples per Leaf (min_samples_leaf=5)"),
    (DecisionTreeClassifier(ccp_alpha=0.02, random_state=42), "Cost Complexity Pruning (ccp_alpha=0.02)")
]
for n_features, n_samples in datasets:
    for clf, strategy_description in strategies:
        train_and_evaluate_on_random_data(n_features, n_samples, clf, strategy_description)
