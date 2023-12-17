import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report

def train_evaluate_knn(X_train, Y_train, X_validation, Y_validation):
    # Train KNN model
    clf = KNeighborsClassifier()
    clf.fit(X_train, Y_train)

    # Predictions
    predictions = clf.predict(X_validation)

    # Evaluation
    accuracy = accuracy_score(Y_validation, predictions) * 100
    mcc = matthews_corrcoef(Y_validation, predictions)
    report = classification_report(Y_validation, predictions)

    # Print results
    print("KNN")
    print("Accuracy:", accuracy)
    print("MCC:", mcc)
    print("Classification Report:\n", report)

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

# Train and evaluate KNN model
train_evaluate_knn(X_train, Y_train, X_validation, Y_validation)
