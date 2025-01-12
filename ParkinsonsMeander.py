import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

# Load the Meander dataset
df1 = pd.read_csv("Meander_HandPD.csv")

# Map categorical values to numeric
mapping_gender = {'M': 0, 'F': 1}
mapping_hand = {'R': 0, 'L': 1}

df1['GENDER'] = df1['GENDER'].map(mapping_gender)
df1['RIGHT/LEFT-HANDED'] = df1['RIGHT/LEFT-HANDED'].map(mapping_hand)

# Define features to scale
meander_features = ['RMS', 'MAX_BETWEEN_ET_HT', 'MIN_BETWEEN_ET_HT','STD_DEVIATION_ET_HT', 'MRT', 'MAX_HT', 'MIN_HT']

# Initialize the scaler
scaler = StandardScaler()

# Scale the selected features
df1[meander_features] = scaler.fit_transform(df1[meander_features])

# Separate features and target
X1 = df1.drop(columns=['CLASS_TYPE', 'IMAGE_NAME'], errors='ignore')

y1 = df1['CLASS_TYPE']

# Split dataset into training and test sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Initialize the RandomForest model
rf_model = RandomForestClassifier(
    random_state=42, max_depth=5, n_estimators=100, min_samples_split=10, min_samples_leaf=4
)
rf_model.fit(X1_train, y1_train)

# Perform 5-fold cross-validation for Meander dataset
cv_scores_meander = cross_val_score(rf_model, X1_train, y1_train, cv=5, scoring='accuracy')

# Print cross-validation results
print(f"Cross-Validation Accuracy Scores (Meander): {cv_scores_meander}")
print(f"Average Cross-Validation Accuracy (Meander): {np.mean(cv_scores_meander):.2f}")

# Save the model
filename = "parkinsons_meandermodel.sav"
pickle.dump(rf_model, open(filename, 'wb'))

# Load the saved model for verification
loaded_model = pickle.load(open(filename, 'rb'))
print("Model loaded successfully.")
