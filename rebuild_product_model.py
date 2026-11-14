"""Re-save the product model and build customer_inference.csv with the CURRENT scikit-learn.

Why: models/product_model.joblib was pickled with scikit-learn 1.6.1 (Colab) and cannot be
loaded by scikit-learn 1.7+ (AttributeError: _RemainderColsList). This script repeats
Patrick's notebook 01 steps 4-6 exactly (same split, same features, same Random Forest) on
data/processed/merged_dataset.csv, so every teammate can load the result.

    python rebuild_product_model.py            # writes models/product_model.joblib and
                                               #        data/processed/customer_inference.csv
    python rebuild_product_model.py --check    # only print metrics, write nothing

Expected held-out metrics (from notebook 01): accuracy 0.3182, weighted F1 0.2934, log loss 1.7182.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
TARGET = "product_category"
NOT_FEATURES = [TARGET, "transaction_id", "customer_id", "purchase_date",
                "purchase_amount", "customer_rating", "review_sentiment"]  # post-purchase / ids


def build():
    merged = pd.read_csv(ROOT / "data/processed/merged_dataset.csv", parse_dates=["purchase_date"])
    fe = merged.copy()
    fe["purchase_month"] = fe.purchase_date.dt.month
    fe["purchase_dayofweek"] = fe.purchase_date.dt.dayofweek
    fe["is_weekend"] = fe.purchase_dayofweek.isin([5, 6]).astype(int)
    fe["engagement_to_interest_ratio"] = fe.engagement_score / (fe.purchase_interest_score + 1e-5)

    # Same grouped split as the notebook: a customer is never in both train and test.
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                               .split(fe, y=fe[TARGET], groups=fe.customer_id))
    train, test = fe.iloc[train_idx].copy(), fe.iloc[test_idx].copy()

    # Customer history is computed from TRAINING rows only (no test leakage).
    agg = train.groupby("customer_id", as_index=False).agg(
        cust_tx_count=("transaction_id", "count"), cust_avg_spend=("purchase_amount", "mean"))
    default_count, default_spend = agg.cust_tx_count.median(), train.purchase_amount.mean()
    train = train.merge(agg, on="customer_id", how="left")
    test = test.merge(agg, on="customer_id", how="left")
    test["cust_tx_count"] = test.cust_tx_count.fillna(default_count)
    test["cust_avg_spend"] = test.cust_avg_spend.fillna(default_spend)

    features = [c for c in train.columns if c not in NOT_FEATURES]
    numeric = train[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical = train[features].select_dtypes(exclude=[np.number]).columns.tolist()
    model = Pipeline([
        ("preprocessor", ColumnTransformer([("num", StandardScaler(), numeric),
                                            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)])),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
    ]).fit(train[features], train[TARGET].astype(str))

    y_test = test[TARGET].astype(str)
    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(test[features])),
        "weighted_f1": f1_score(y_test, model.predict(test[features]), average="weighted", zero_division=0),
        "log_loss": log_loss(y_test, model.predict_proba(test[features]), labels=model.classes_),
    }

    # One pre-visit row per customer: their latest visit's time features + history.
    latest = fe.sort_values("purchase_date").groupby("customer_id", as_index=False).last()
    latest = latest.merge(agg, on="customer_id", how="left")
    latest["cust_tx_count"] = latest.cust_tx_count.fillna(default_count)
    latest["cust_avg_spend"] = latest.cust_avg_spend.fillna(default_spend)
    inference = latest[["customer_id"] + features].sort_values("customer_id").reset_index(drop=True)
    return model, features, inference, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="print metrics only, write nothing")
    args = parser.parse_args()
    model, features, inference, metrics = build()
    print("Held-out metrics:", {k: round(v, 4) for k, v in metrics.items()})
    print("Random-chance accuracy for 5 classes: 0.2000")
    meta = json.loads((ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
    expected = meta["product"]["feature_columns"]
    assert features == expected, f"Feature columns differ from model_metadata.json:\n{features}\n{expected}"
    if args.check:
        return
    joblib.dump(model, ROOT / "models/product_model.joblib")
    inference.to_csv(ROOT / "data/processed/customer_inference.csv", index=False)
    print(f"Saved models/product_model.joblib and data/processed/customer_inference.csv ({len(inference)} customers)")


if __name__ == "__main__":
    main()
