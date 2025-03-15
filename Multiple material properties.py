# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen import Composition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

# Step 1: Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/materialsvirtuallab/matminer/master/matminer/datasets/materials_project_multitarget.csv")

# Display the first few rows of the dataset
print(data.head())

# Step 2: Data Preprocessing
# Convert composition strings to pymatgen Composition objects
data = StrToComposition().featurize_dataframe(data, "formula")

# Step 3: Feature Engineering
# Use ElementProperty to generate composition-based features
ep_featurizer = ElementProperty.from_preset("magpie")
data = ep_featurizer.featurize_dataframe(data, col_id="composition")

# Drop rows with missing values
data = data.dropna()

# Step 4: Define Features and Targets
# Features: All elemental property features
# Targets: Multiple material properties (e.g., band_gap, formation_energy_per_atom, bulk_modulus)
X = data.drop(columns=["formula", "composition", "band_gap", "formation_energy_per_atom", "bulk_modulus"])
y = data[["band_gap", "formation_energy_per_atom", "bulk_modulus"]]

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Multi-Target Machine Learning Model
# Use MultiOutputRegressor with Random Forest Regressor
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics for each target
for i, target in enumerate(y.columns):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"Target: {target}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print()

# Step 8: Visualize Results
# Plot actual vs predicted values for each target
for i, target in enumerate(y.columns):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5)
    plt.plot([min(y_test.iloc[:, i]), max(y_test.iloc[:, i])], 
             [min(y_test.iloc[:, i]), max(y_test.iloc[:, i])], color="red", linestyle="--")
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"Actual vs Predicted {target}")
    plt.show()

# Plot feature importance (for the first target as an example)
feature_importance = pd.Series(model.estimators_[0].feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).head(20).plot(kind="bar", figsize=(10, 6))
plt.title("Top 20 Feature Importances for Bandgap Prediction")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
