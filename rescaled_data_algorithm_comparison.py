import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn import tree
import graphviz  # Make sure to install Graphviz: https://graphviz.gitlab.io/download/

# Load the dataset
url = "data.csv"
features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
            "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2",
            "D2", "PPE", "status"]

dataset = pd.read_csv(url, names=features)

# Prepare the dataset
array = dataset.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(array)
X = scaled[:, 0:22]
Y = scaled[:, 22]
validation_size = 0.25
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# 10-fold cross-validation
num_folds = 10
scoring = 'accuracy'

# Algorithms / Models
models = [('DT', DecisionTreeClassifier())]

# Evaluate each algorithm / model
print("Scores for each algorithm:")
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    print(f"{name}: {cv_results.mean() * 100:.2f}% (+/- {cv_results.std() * 100:.2f}%)")

# Train and evaluate on the validation set
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(f"{name} on validation set: {accuracy_score(Y_validation, predictions) * 100:.2f}%")
    print(f"MCC: {matthews_corrcoef(Y_validation, predictions)}")

    # Export the decision tree visualization
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=features[:-1], class_names=['0', '1'],
                                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree_visualization")
