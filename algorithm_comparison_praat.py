import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load the dataset
url = "data.csv"
features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
            "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "status"]

dataset = pd.read_csv(url, names=features)

# Prepare the dataset
X = dataset.iloc[:, 0:16]
Y = dataset.iloc[:, 16]
validation_size = 0.3
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# 10-fold cross-validation
num_folds = 10
scoring = 'accuracy'

# Algorithms / Models
models = [('LR', LogisticRegression()),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()),
          ('DT', DecisionTreeClassifier()),
          ('NN', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100,))),
          ('NB', GaussianNB()),
          ('GB', GradientBoostingClassifier(n_estimators=10000))]

# Evaluate each algorithm / model
print("Scores for each algorithm:")
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    print(f"{name}: {cv_results.mean() * 100:.2f}% (+/- {cv_results.std() * 100:.2f}%)")

# Train and evaluate on validation set
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(f"{name} on validation set: {accuracy_score(Y_validation, predictions) * 100:.2f}%")
    print(f"MCC: {matthews_corrcoef(Y_validation, predictions)}\n")
