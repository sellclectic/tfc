"""
Inference demo for Project S.H.O.E.S.
AI-powered footwear lifecycle optimization platform.

Loads trained models and predicts the optimal lifecycle path and
estimated resale value for a set of demonstration shoes.

Usage:
    python predict.py

    Requires model_classifier.pkl and model_regressor.pkl to exist.
    Run train.py first if they are missing.

Extension point:
    DEMO_SHOES is a plain list of dicts. To accept user input, replace
    this list with an argparse-based input or a CSV reader.
"""

import os

import joblib
import pandas as pd

from data import BRAND_MULTIPLIERS, DEFAULT_POLICY, build_shoe_for_material
from main import recyclability_score

CLASSIFIER_PATH = "model_classifier.pkl"
REGRESSOR_PATH  = "model_regressor.pkl"

# ---------------------------------------------------------------------------
# Demo inputs — 5 shoes exercising all three predicted paths
# ---------------------------------------------------------------------------

DEMO_SHOES = [
    # Clear resale: premium sneaker, excellent condition, high demand
    dict(
        label="Premium sneaker (near-new)",
        category="sneaker", material="leather", brand_tier="premium",
        condition_score=0.92, age_months=8, original_price=280.0,
        resale_market_demand=0.85,
    ),
    # Clear repair: quality boot, mid condition, below resale threshold
    dict(
        label="Mid-tier boot (worn, repairable)",
        category="boot", material="mixed", brand_tier="mid",
        condition_score=0.52, age_months=36, original_price=160.0,
        resale_market_demand=0.45,
    ),
    # Edge case: good condition but weak demand — may tip to repair
    dict(
        label="Fast-fashion sneaker (decent shape, low demand)",
        category="sneaker", material="synthetic", brand_tier="fast",
        condition_score=0.68, age_months=18, original_price=75.0,
        resale_market_demand=0.22,
    ),
    # Clear recycle: old sandal, very poor condition
    dict(
        label="Sandal (end of life)",
        category="sandal", material="synthetic", brand_tier="fast",
        condition_score=0.12, age_months=60, original_price=45.0,
        resale_market_demand=0.10,
    ),
    # Edge case: high-value, older premium boot still worth repairing
    dict(
        label="Premium boot (aged, still valuable)",
        category="boot", material="leather", brand_tier="premium",
        condition_score=0.58, age_months=48, original_price=420.0,
        resale_market_demand=0.40,
    ),
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    clf_path: str = CLASSIFIER_PATH,
    reg_path: str = REGRESSOR_PATH,
) -> tuple:
    """
    Load and return (clf_pipeline, reg_pipeline).

    Raises FileNotFoundError with an actionable message if either file
    is missing — the most common error on a fresh clone.
    """
    for path in [clf_path, reg_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file '{path}' not found. "
                "Run 'python train.py' first to train and save models."
            )
    return joblib.load(clf_path), joblib.load(reg_path)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_feature_row(shoe_dict: dict) -> dict:
    """
    Takes a shoe input dict (without recyclability_score) and returns a
    full feature dict including recyclability_score computed via the same
    rule-based logic used during training (main.recyclability_score).

    This ensures feature construction is identical between training and
    inference — no distribution mismatch.
    """
    shoe = build_shoe_for_material(shoe_dict["material"])
    rec_score = recyclability_score(shoe, DEFAULT_POLICY)

    return {
        "category":             shoe_dict["category"],
        "material":             shoe_dict["material"],
        "brand_tier":           shoe_dict["brand_tier"],
        "condition_score":      shoe_dict["condition_score"],
        "age_months":           shoe_dict["age_months"],
        "original_price":       shoe_dict["original_price"],
        "resale_market_demand": shoe_dict["resale_market_demand"],
        "recyclability_score":  rec_score,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_predictions(
    clf_pipeline,
    reg_pipeline,
    shoes: list,
) -> pd.DataFrame:
    """
    Runs both models on a list of shoe dicts.
    Returns a DataFrame combining input features with predictions.
    """
    rows = [build_feature_row(s) for s in shoes]
    X = pd.DataFrame(rows)

    predicted_paths  = clf_pipeline.predict(X)
    predicted_values = reg_pipeline.predict(X)

    results = pd.DataFrame(shoes).copy()
    results["predicted_path"]  = predicted_paths
    results["predicted_value"] = predicted_values

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_output(results: pd.DataFrame) -> None:
    header = (
        f"\n{'=== Project S.H.O.E.S. — Lifecycle Path Predictor':=<70}\n"
    )
    print(header)

    col_w = {"label": 38, "path": 9, "value": 12}
    sep = "-" * 70

    print(
        f"  {'Shoe':<{col_w['label']}}  {'Path':<{col_w['path']}}  {'Est. Resale':>{col_w['value']}}"
    )
    print(sep)

    for _, row in results.iterrows():
        path  = row["predicted_path"].upper()
        value = f"${row['predicted_value']:.2f}"
        print(
            f"  {row['label']:<{col_w['label']}}  {path:<{col_w['path']}}  {value:>{col_w['value']}}"
        )

    print(sep)
    print(
        "\n  Note: estimated resale value shown for all shoes as a market\n"
        "  reference. Near-zero values on RECYCLE predictions are expected.\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    clf_pipeline, reg_pipeline = load_models()
    results = run_predictions(clf_pipeline, reg_pipeline, DEMO_SHOES)
    format_output(results)


if __name__ == "__main__":
    main()
