import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Data Ingestion
# We use the Portuguese "Vinho Verde" dataset (Red Wine) directly from UCI
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
url = "./winequality-red.csv"

# Note: This dataset uses semicolons ';' as separators, not commas
print(f"Downloading data from {url}...")
df = pd.read_csv(url, sep=';')

# 2. Preprocessing
# Target variable is 'quality' (score between 0 and 10)
X = df.drop('quality', axis=1)
y = df['quality']

# Split data (Optional for the artifact, but good practice to evaluate)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
# RandomForest is robust and works well without heavy scaling, perfect for a demo.
print("Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation (Just to prove it works)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Model Training Complete. MSE: {mse:.4f}")

# 5. Serialization (The crucial step for Cloud Deployment)
model_filename = 'wine_quality_model.joblib'
joblib.dump(model, model_filename)

print(f"Success! Model saved to '{model_filename}'.")
print("You can now move this file into your Docker container.")