import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

data = pd.read_csv('data/student_performance_dataset.csv')

# Print object columns for debugging
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
print("Columns with object dtype:", categorical_cols)

# Remove the target column if it's in the list
if 'performance' in categorical_cols:
    categorical_cols.remove('performance')

# One-hot encode all remaining categorical columns
if categorical_cols:
    data = pd.get_dummies(data, columns=categorical_cols)

X = data.drop(columns=['performance'])
y = data['performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

os.makedirs('models', exist_ok=True)

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")