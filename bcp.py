import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv('breast-cancer.csv')


df = df.drop('id', axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Split features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]


# EVALUATION

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("LOGISTIC REGRESSION - BREAST CANCER DETECTION")
print("=" * 50)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"True Negatives (Benign correct): {cm[0,0]}")
print(f"False Positives (Benign wrong): {cm[0,1]}")
print(f"False Negatives (Cancer wrong): {cm[1,0]}")
print(f"True Positives (Cancer correct): {cm[1,1]}")


# SIGMOID DECISION BOUNDARY

plt.figure(figsize=(12, 7))

# Use the most important feature for x-axis (radius_mean has highest coefficient)
feature_idx = 0  # radius_mean
feature_name = X.columns[feature_idx]


X_feature = X_test_scaled[:, feature_idx]
sorted_idx = np.argsort(X_feature)
X_sorted = X_feature[sorted_idx]
y_sorted_proba = y_pred_proba[sorted_idx]

# Plot the sigmoid curve
plt.plot(X_sorted, y_sorted_proba, 'b-', linewidth=2, label='Logistic Regression (Sigmoid)')

# Plot actual data points
scatter = plt.scatter(X_feature, y_test, c=y_test, cmap='RdYlGn_r', 
                      alpha=0.6, s=50, edgecolors='black', linewidth=1)

# Add decision boundary line
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')

# Customize plot
plt.xlabel(f'{feature_name} (Standardized)', fontsize=12)
plt.ylabel('Probability of Malignancy', fontsize=12)
plt.title(f'Logistic Regression Sigmoid Curve\nBreast Cancer Detection Accuracy: {accuracy:.2%}', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter)
cbar.set_label('Actual Diagnosis', rotation=270, labelpad=15)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Benign (0)', 'Malignant (1)'])

plt.text(0.02, 0.95, f'Model: Logistic Regression\nSamples: {len(X_test)}\nFeatures: {X.shape[1]}', 
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# PREDICTION FUNCTION

def predict_cancer(patient_data):
    """Predict if tumor is cancerous"""
    scaled_data = scaler.transform([patient_data])
    prob = model.predict_proba(scaled_data)[0][1]
    prediction = "MALIGNANT (Cancer)" if prob >= 0.5 else "BENIGN (No Cancer)"
    return prediction, prob

print("\n" + "=" * 50)
print("SAMPLE PREDICTION")
print("=" * 50)

# Test with first sample from test set
sample = X_test.iloc[0].values
true_label = "Malignant" if y_test.iloc[0] == 1 else "Benign"
pred, prob = predict_cancer(sample)

print(f"True Diagnosis: {true_label}")
print(f"Prediction: {pred}")
print(f"Probability of Cancer: {prob:.3f} ({prob*100:.1f}%)")