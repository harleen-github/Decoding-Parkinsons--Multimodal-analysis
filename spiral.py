import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

# Load the Spiral dataset
df2 = pd.read_csv("Spiral_HandPD.csv")

# Map categorical values to numeric
mapping_gender = {'M': 0, 'F': 1}
mapping_hand = {'R': 0, 'L': 1}

df2['GENDER'] = df2['GENDER'].map(mapping_gender)
df2['RIGHT/LEFT-HANDED'] = df2['RIGHT/LEFT-HANDED'].map(mapping_hand)

# Define features to scale
spiral_features = ['RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT',
                   'STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT']

# Initialize the scaler
scaler = StandardScaler()

# Scale the selected features
df2[spiral_features] = scaler.fit_transform(df2[spiral_features])

# Separate features and target
X2 = df2.drop(columns=['CLASS_TYPE', 'IMAGE_NAME'], errors='ignore')

y2 = df2['CLASS_TYPE']

# Split dataset into training and test sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

# Initialize the RandomForest model
rf_model = RandomForestClassifier(
    random_state=42, max_depth=5, n_estimators=100, min_samples_split=10, min_samples_leaf=4
)
rf_model.fit(X2_train, y2_train)

# Perform 5-fold cross-validation for Spiral dataset
cv_scores_spiral = cross_val_score(rf_model, X2_train, y2_train, cv=5, scoring='accuracy')

# Print cross-validation results
print(f"Cross-Validation Accuracy Scores (Spiral): {cv_scores_spiral}")
print(f"Average Cross-Validation Accuracy (Spiral): {np.mean(cv_scores_spiral):.2f}")

# Save the model
filename = "parkinsons_spiralmodel.sav"
pickle.dump(rf_model, open(filename, 'wb'))

# Load the saved model for verification
loaded_model = pickle.load(open(filename, 'rb'))
print("Model loaded successfully.")
