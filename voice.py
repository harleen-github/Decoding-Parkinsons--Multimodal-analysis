# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report  # Import classification_report
import pickle

# Load and preprocess data
df = pd.read_csv("parkinsons voice oxford.data", delimiter=",")
print(df.columns)
# df.columns = df.columns.str.replace(" ", "_").str.lower()
df = df.drop_duplicates()

# Check for numeric columns
numeric_cols = df.select_dtypes(include='number')

# Prepare features and target
X = numeric_cols.drop(columns=['status'], errors='ignore')
y = df['status']
print(df.columns)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(
    random_state=42, 
    max_depth=5, 
    n_estimators=100, 
    min_samples_split=10, 
    min_samples_leaf=4
)

# Fit the model
rf_model.fit(X_train, y_train)

# Calculate cross-validation scores
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

# Print cross-validation results
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Average Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")

# Save the trained model to a file
filename = "parkinsons_voicemodel.sav"
pickle.dump(rf_model, open(filename, 'wb'))

# Load the saved model for verification
loaded_model = pickle.load(open(filename, 'rb'))
print("Model loaded successfully.")

# Test the loaded model with a sample input
sample_features = [
    119.992, 157.302, 74.997, 0.00784, 0.00007,
    0.0037, 0.00554, 0.01109, 0.04374, 0.426,
    0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
    0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654
]

# Make a prediction
prediction = loaded_model.predict([sample_features])
print(f"Prediction for the sample features: {prediction}")

# Predict labels on the test set
y_pred = loaded_model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))