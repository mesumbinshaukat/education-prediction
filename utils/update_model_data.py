import os
import pickle
import pandas as pd

def update_model_data(input_data, feature_columns, model_path='models/best_model.pkl'):
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    # New data entry as DataFrame
    df_new = pd.DataFrame([input_data], columns=feature_columns)

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        if hasattr(model, 'X') and isinstance(model.X, pd.DataFrame):
            model.X = pd.concat([model.X, df_new], ignore_index=True)
        else:
            model.X = df_new

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print("Model updated successfully.")

    except Exception as e:
        print(f"Failed to update model: {e}")
