from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_evaluate(clf, X_train, X_test, y_train, y_test, feature_names, strategy_description):
    clf.fit(X_train, y_train)
    rules = export_text(clf, feature_names=feature_names)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nStrategy: {strategy_description}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Number of Rules: {rules.count('class')}")
    print("Rules:\n", rules)
    print("-" * 50)

# Load the Wine dataset
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
feature_names = load_wine().feature_names

# Strategy 1: Limited Tree Depth
clf_depth = DecisionTreeClassifier(max_depth=3, random_state=42)
train_and_evaluate(clf_depth, X_train, X_test, y_train, y_test, feature_names, "Limited Tree Depth (max_depth=3)")

# Strategy 2: Increased Min Samples per Leaf
clf_samples_leaf = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
train_and_evaluate(clf_samples_leaf, X_train, X_test, y_train, y_test, feature_names, "Increased Min Samples per Leaf (min_samples_leaf=5)")

# Strategy 3: Cost Complexity Pruning
clf_ccp_alpha = DecisionTreeClassifier(ccp_alpha=0.02, random_state=42) # Adjusted for Wine dataset complexity
train_and_evaluate(clf_ccp_alpha, X_train, X_test, y_train, y_test, feature_names, "Cost Complexity Pruning (ccp_alpha=0.02)")
