
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import joblib

# Model loading function
def model_fn(model_dir):
    """
    Load the trained model from the specified directory.

    Args:
        model_dir (str): Directory where the model artifact is stored.

    Returns:
        model: Loaded model object.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")  # Default SageMaker training input path
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")        # Default SageMaker model path
    args = parser.parse_args()

    # Load training data
    train_file = os.path.join(args.train, "train_data.csv")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data file not found at {train_file}")
    
    train_data = pd.read_csv(train_file, header=None)
    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    # Train the model
    svc_model = SVC()  # Default parameters; you can tune this if needed
    svc_model.fit(x_train, y_train)

    # Save the model
    model_path = os.path.join(args.model_dir, "model.joblib")
    os.makedirs(args.model_dir, exist_ok=True)  # Ensure the model directory exists
    joblib.dump(svc_model, model_path)
    print(f"Model successfully saved at {model_path}")
    