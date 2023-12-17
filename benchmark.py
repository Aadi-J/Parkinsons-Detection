import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score

# Load the dataset
url = "data.csv"
features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
            "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3",
            "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2",
            "D2", "PPE", "status"]

dataset = pd.read_csv(url, names=features)

# Prepare the dataset
X = dataset.iloc[:, 0:22]
Y = dataset.iloc[:, 22]
validation_size = 0.3
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Dummy predictions for benchmark
benchmark_predictions = [1] * len(X_validation)

# Evaluate on the validation set
benchmark_accuracy = accuracy_score(Y_validation, benchmark_predictions) * 100
benchmark_mcc = matthews_corrcoef(Y_validation, benchmark_predictions)

print(f"Benchmark on validation set: {benchmark_accuracy:.2f}%")
print(f"MCC: {benchmark_mcc}\n")
