"""
Training pipeline for Project S.H.O.E.S.
AI-powered footwear lifecycle optimization platform.

Trains two models:
    1. RandomForestClassifier  → predicts optimal_path (resale / repair / recycle)
    2. RandomForestRegressor   → predicts resale_value (continuous)

Both models share a ColumnTransformer preprocessor and are saved as
Pipeline objects so predict.py can call pipeline.predict() directly on
raw DataFrames without a separate preprocessing step.

Usage:
    PYTHONPATH=src python -m shoes.common.train
"""

import os

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shoes.common.data import generate_dataset

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = ["category", "material", "brand_tier"]
NUMERICAL_FEATURES = [
    "condition_score",
    "age_months",
    "original_price",
    "resale_market_demand",
    "recyclability_score",
]

CLASSIFIER_PATH = "model_classifier.pkl"
REGRESSOR_PATH  = "model_regressor.pkl"


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Returns the shared feature preprocessor used by both pipelines.

    OneHotEncoder settings:
        handle_unknown="ignore"  — returns all-zero columns for unseen
                                   categories at inference time instead of
                                   raising an exception.
        sparse_output=False      — dense output avoids deprecation warnings
                                   in scikit-learn 1.4+ at this data size.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERICAL_FEATURES),
        ]
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(X_train, y_train, preprocessor: ColumnTransformer) -> Pipeline:
    """
    Builds and fits the classification pipeline.

    class_weight="balanced" compensates for the repair-dominant class
    distribution in the synthetic data, improving F1 macro on minority
    classes (resale, recycle).
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_regressor(X_train, y_train, preprocessor: ColumnTransformer) -> Pipeline:
    """Builds and fits the regression pipeline."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }


def evaluate_regressor(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {"rmse": rmse}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating dataset...")
    df = generate_dataset()

    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X       = df[feature_cols]
    y_class = df["optimal_path"]
    y_reg   = df["resale_value"]

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = (
        train_test_split(
            X, y_class, y_reg,
            test_size=0.2,
            stratify=y_class,
            random_state=42,
        )
    )

    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"Class distribution (train):\n{y_class_train.value_counts().to_string()}\n")

    # Train classifier
    print("Training classifier...")
    preprocessor = build_preprocessor()
    clf_pipeline = train_classifier(X_train, y_class_train, preprocessor)

    # Train regressor (fresh preprocessor instance to avoid fitted-state sharing)
    print("Training regressor...")
    reg_pipeline = train_regressor(X_train, y_reg_train, build_preprocessor())

    # Evaluate
    clf_metrics = evaluate_classifier(clf_pipeline, X_test, y_class_test)
    reg_metrics = evaluate_regressor(reg_pipeline, X_test, y_reg_test)

    print("\n=== Classification Report ===")
    print(f"Accuracy : {clf_metrics['accuracy']:.4f}")
    print(f"F1 Macro : {clf_metrics['f1_macro']:.4f}")

    print("\n=== Regression Report ===")
    print(f"RMSE     : {reg_metrics['rmse']:.2f}")

    # Save models
    for path in [CLASSIFIER_PATH, REGRESSOR_PATH]:
        if os.path.exists(path):
            print(f"\nWarning: overwriting existing {path}")

    joblib.dump(clf_pipeline, CLASSIFIER_PATH)
    joblib.dump(reg_pipeline, REGRESSOR_PATH)

    print(f"\nModels saved:")
    print(f"  {CLASSIFIER_PATH}")
    print(f"  {REGRESSOR_PATH}")


if __name__ == "__main__":
    main()
