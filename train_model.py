import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
from skimage.io import imread_collection
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, hog
from skimage.transform import resize
from skimage.exposure import equalize_hist
from scipy.stats import kurtosis, skew

def extract_features(image):
    # Resize image to a fixed size
    image = resize(image, (256, 256), anti_aliasing=True)
    gray = rgb2gray(image)
    
    # Apply histogram equalization
    gray_eq = equalize_hist(gray)
    
    # Extract LBP features
    lbp = local_binary_pattern(gray_eq, P=8, R=1)
    
    # Extract HOG features
    hog_feat = hog(gray_eq, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    
    # Calculate statistical features
    variance = np.var(gray_eq)
    skewness = skew(gray_eq.flatten())
    curtosis = kurtosis(gray_eq.flatten())
    lbp_mean = np.mean(lbp)
    lbp_var = np.var(lbp)
    
    return np.concatenate(([variance, skewness, curtosis, lbp_mean, lbp_var], hog_feat))

# Load and process images
authentic_path = 'data/authentic/*.jpg'
counterfeit_path = 'data/counterfeit/*.jpg'

authentic_images = imread_collection('C:/Users/samee/OneDrive/Desktop/FINAL/Bank-Note-Authentication/data/authentic/*.jpg')
counterfeit_images = imread_collection('C:/Users/samee/OneDrive/Desktop/FINAL/Bank-Note-Authentication/data/counterfeit/*.jpg')

print(f"Number of authentic images: {len(authentic_images)}")
print(f"Number of counterfeit images: {len(counterfeit_images)}")

authentic_features = [extract_features(img) for img in authentic_images]
counterfeit_features = [extract_features(img) for img in counterfeit_images]

# Create DataFrame
data = pd.DataFrame(authentic_features + counterfeit_features)
data['class'] = [1] * len(authentic_features) + [0] * len(counterfeit_features)

print(f"Shape of the dataset: {data.shape}")

# Split features and target
X = data.drop('class', axis=1)
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline for the model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest set accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the best model
joblib.dump(best_model, 'banknote_auth_model.joblib')

print(f"\nBest model saved as 'banknote_auth_model.joblib'")
print(f"Best model parameters: {grid_search.best_params_}")

