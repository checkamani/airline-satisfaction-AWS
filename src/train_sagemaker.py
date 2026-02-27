import argparse
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    args = parser.parse_args()

    files = [f for f in os.listdir(args.data_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV found in {args.data_dir}. Found: {os.listdir(args.data_dir)}")

    csv_path = os.path.join(args.data_dir, files[0])
    df = pd.read_csv(csv_path)

    df = df.drop(columns=["Unnamed: 0", "id"], errors="ignore")
    df["satisfaction"] = df["satisfaction"].map({"satisfied": 1, "neutral or dissatisfied": 0})

    if df["satisfaction"].isna().any():
        raise ValueError("Target mapping produced NaNs. Check satisfaction values.")

    X = df.drop(columns=["satisfaction"])
    y = df["satisfaction"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ]
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=300))
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs(args.model_dir, exist_ok=True)
    out_path = os.path.join(args.model_dir, "model.pkl")
    joblib.dump(clf, out_path)
    print("Saved model to:", out_path)

if __name__ == "__main__":
    main()
