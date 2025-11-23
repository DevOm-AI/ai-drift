import pandas as pd
from sklearn.model_selection import train_test_split

def load_reference_dataset(path: str):
    """Loads the reference dataset for model training."""
    df = pd.read_csv(path)
    return df

def prepare_features(df: pd.DataFrame):
    """Splits dataset into X (features) and y (label)."""
    X = df.drop("purchased", axis=1)
    y = df["purchased"]
    return X, y

if __name__ == "__main__":
    df = load_reference_dataset("src/ai_drift/data/reference.csv")
    X, y = prepare_features(df)
    print("Dataset loaded!")
    print(X.head())
