import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import pickle
import os

client = MongoClient("mongodb://localhost:27017/")
db = client['your_database_name']  # Replace with your DB
collection = db.predictionHistory

data = list(collection.find())
if not data:
    print("❌ No data found.")
    exit()

df = pd.DataFrame(data)
df.drop(columns=['_id'], inplace=True, errors='ignore')
df.dropna(inplace=True)

X = df[['attendance', 'homework_completion', 'test_scores']]
y = df['prediction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

os.makedirs('models', exist_ok=True)
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model & Scaler retrained and saved.")
