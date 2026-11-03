
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the real dataset
data = pd.read_csv(
    "dataset/landslide_modelling_dataset_12409.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# 2. Select input features
features = [
    "Aspect_z",
    "Dist_River_z",
    "Elevation_z",
    "FlowAccumulation_z",
    "NDVI_z",
    "Rainfall_z",
    "Slope_z",
    "LULC_10",
    "LULC_20",
    "LULC_30",
    "LULC_40",
    "LULC_50",
    "LULC_60",
    "LULC_70",
    "LULC_80",
    "LULC_90",
    "LULC_100"
]

X = data[features]
y = data["class"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

# 5. Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(y_test, predictions)

print("\nModel Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(matrix)

# 8. Save the trained model
joblib.dump(model, "landslide_model.pkl")

# 9. Save evaluation results
evaluation_results = {
    "accuracy": accuracy,
    "report": report,
    "confusion_matrix": matrix,
    "classes": model.classes_,
    "feature_names": features
}

joblib.dump(
    evaluation_results,
    "evaluation_results.pkl"
)

print("\nModel saved successfully!")
print("Evaluation results saved successfully!")
