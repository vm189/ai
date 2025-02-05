import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Dataset
data = load_iris()
X, y = data.data, data.target

# Step 2: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the Model
model = RandomForestClassifier(random_state=42)

# Step 4: Define the Search Space for Hyperparameters
param_space = {
    "n_estimators": (10, 200),          # Number of trees in the forest
    "max_depth": (1, 20),              # Maximum depth of each tree
    "min_samples_split": (2, 10),      # Minimum samples to split a node
    "min_samples_leaf": (1, 10),       # Minimum samples at each leaf
    "max_features": ["sqrt", "log2", None]  # Number of features considered for split
}

# Step 5: Use Bayesian Optimization for Hyperparameter Tuning
optimizer = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=30,  # Number of iterations to search
    cv=3,       # 3-fold cross-validation
    random_state=42,
    n_jobs=-1
)

# Step 6: Train the Optimized Model
print("Starting Bayesian Optimization...")
optimizer.fit(X_train, y_train)

# Step 7: Evaluate the Best Model
best_model = optimizer.best_estimator_
y_pred = best_model.predict(X_test)

print("\nBest Parameters:", optimizer.best_params_)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Save the Best Model
import joblib
joblib.dump(best_model, "optimized_rf_model.pkl")
print("\nModel saved as 'optimized_rf_model.pkl'.")
