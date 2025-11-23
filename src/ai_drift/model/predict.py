import pandas as pd
import joblib


def load_model(path="artifacts/model.joblib"):
    """Loads the trained ML model (preprocessing + classifier)."""
    model = joblib.load(path)
    return model


def predict(model, input_df: pd.DataFrame):
    """Predicts output for given dataframe."""
    preds = model.predict(input_df)
    return preds


if __name__ == "__main__":
    # Example: run prediction using drifted data
    model = load_model()
    print("Model loaded successfully!")

    # Change this filename to test other datasets
    df = pd.read_csv("src/ai_drift/data/current_seasonal.csv")

    preds = predict(model, df.drop("purchased", axis=1))
    print("Predictions:", preds[:10])
