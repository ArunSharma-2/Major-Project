#------------------------------IMPORTS------------------------------#

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#------------------------------DATA COLLECTION------------------------------#

df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

#------------------------------MODEL TRAINING------------------------------#

class_distribution = y.value_counts().to_dict()
print("Class distribution:")
print(class_distribution)
print("Imbalance ratio (majority/minority):", round(max(class_distribution.values()) / min(class_distribution.values()), 2))

logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    ),
)

random_forest_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
)

logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

#------------------------------MODEL EVALUATION------------------------------#

logistic_predictions = logistic_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_predictions)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)

#------------------------------CONFUSION MATRIX & CLASSIFICATION REPORT------------------------------#

logistic_confusion = confusion_matrix(y_test, logistic_predictions)
random_forest_confusion = confusion_matrix(y_test, random_forest_predictions)

logistic_false_negatives = logistic_confusion[1, 0]
random_forest_false_negatives = random_forest_confusion[1, 0]

#------------------------------BUSINESS RISK INTERPRETATION------------------------------#

print("Logistic Regression")
print("Accuracy:", logistic_accuracy)
print("\nConfusion Matrix:")
print(logistic_confusion)
print("\nClassification Report:")
print(classification_report(y_test, logistic_predictions, zero_division=0))

print("\nRandom Forest")
print("Accuracy:", random_forest_accuracy)
print("\nConfusion Matrix:")
print(random_forest_confusion)
print("\nClassification Report:")
print(classification_report(y_test, random_forest_predictions, zero_division=0))

print("\nBusiness Risk Interpretation")
print("- The dataset is highly imbalanced, so accuracy alone can be misleading.")
print("- High false negatives mean more fraud cases may be missed.")
print("- A lower false-negative rate means lower exposure to financial loss.")
print("- The model with fewer missed fraud cases is better for risk control.")
print(f"- Logistic Regression missed {logistic_false_negatives} fraud cases.")
print(f"- Random Forest missed {random_forest_false_negatives} fraud cases.")
print("- Class weights and balanced sampling were used to reduce the impact of imbalance.")

if random_forest_false_negatives < logistic_false_negatives:
    print("- Random Forest presents lower business risk because it misses fewer fraud cases.")
elif random_forest_false_negatives > logistic_false_negatives:
    print("- Logistic Regression presents lower business risk because it misses fewer fraud cases.")
else:
    print("- Both models show similar business risk for missed fraud cases.")

#------------------------------MODEL COMPARISON------------------------------#

print("\nComparison")
print(f"Logistic Regression Accuracy: {logistic_accuracy:.4f}")
print(f"Random Forest Accuracy: {random_forest_accuracy:.4f}")

if random_forest_accuracy > logistic_accuracy:
    print("Random Forest performed better.")
elif random_forest_accuracy < logistic_accuracy:
    print("Logistic Regression performed better.")
else:
    print("Both models performed equally well.")

#------------------------------VISUALIZATION------------------------------#

labels = ["Logistic Regression", "Random Forest"]
values = [logistic_accuracy, random_forest_accuracy]

plt.figure(figsize=(7, 4))
plt.bar(labels, values, color=["#4C78A8", "#F58518"])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1.0)
plt.xticks(rotation=0)
for index, value in enumerate(values):
    plt.text(index, value + 0.01, f"{value:.4f}", ha="center")
plt.tight_layout()
plt.savefig("images/model_comparison.png")
plt.show()