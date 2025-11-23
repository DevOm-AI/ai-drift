import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

import joblib
import os


def load_reference_dataset(path: str):
    """Loads the reference dataset for model training."""
    df = pd.read_csv(path)
    return df

def prepare_features(df: pd.DataFrame):
    """Splits dataset into X (features) and y (label)."""
    X = df.drop("purchased", axis=1)
    y = df["purchased"]
    return X, y


def build_preprocessing_pipeline():
    """Creates a preprocessing pipeline for numeric and categorical features."""
    
    numeric_features = ["age", "income", "spend_score", "membership_years"]
    categorical_features = ["city"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_model(preprocessor, X, y):
    """Builds a training pipeline and trains the model."""
    
    model = LogisticRegression(max_iter=1000)

    # Create full pipeline: preprocessing + model
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    clf.fit(X, y)
    return clf


def save_model(model, path="artifacts/model.joblib"):
    """Saves the trained ML model (pipeline) to disk."""
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    df = load_reference_dataset("src/ai_drift/data/reference.csv")
    X, y = prepare_features(df)

    print("Dataset loaded!")
    print(X.head())

    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(X)
    print("Preprocessing pipeline built successfully!")

    clf = train_model(preprocessor, X, y)
    print("Model trained successfully!")

    save_model(clf)