"""
Synthetic dataset generator for Project S.H.O.E.S.
AI-powered footwear lifecycle optimization platform.

Generates structured footwear records with labeled lifecycle outcomes
(resale / repair / recycle) and estimated resale values.

Integration note:
    recyclability_score is computed by calling main.recyclability_score()
    via build_shoe_for_material(). This reuses the rule-based logic from
    the existing prototype rather than duplicating it.

Assumption:
    All rows use a single DEFAULT_POLICY (mirrors Boston recycling rules
    from main.py). This is a known simplification — only polyester is
    accepted, yielding three distinct recyclability_score values tied to
    the material categorical feature.
"""

import numpy as np
import pandas as pd

from main import RecyclingPolicy, Shoe, recyclability_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42

# Material compositions aligned with material names defined in main.py
# (rubber, polyester, eva_foam, leather).
MATERIAL_COMPOSITIONS = {
    "leather":   {"leather": 0.50, "rubber": 0.30, "polyester": 0.20},
    "synthetic": {"polyester": 0.60, "eva_foam": 0.25, "rubber": 0.15},
    # Mixed yields a mid-range recyclability_score (~0.35) to give the
    # model a gradient between leather (0.20) and synthetic (0.60).
    "mixed":     {"leather": 0.25, "polyester": 0.35, "eva_foam": 0.20, "rubber": 0.20},
}

# Mirrors the Boston recycling policy in main.py: only polyester accepted.
DEFAULT_POLICY = RecyclingPolicy(
    city="Default",
    accepted_materials={
        "rubber": False,
        "polyester": True,
        "eva_foam": False,
        "leather": False,
    },
)

BRAND_MULTIPLIERS = {"premium": 1.1, "mid": 0.9, "fast": 0.6}


# ---------------------------------------------------------------------------
# Bridge function
# ---------------------------------------------------------------------------

def build_shoe_for_material(material: str) -> Shoe:
    """
    Returns a synthetic Shoe whose material composition corresponds to the
    categorical material value. Used to compute recyclability_score via the
    rule-based logic in main.py.
    """
    composition = MATERIAL_COMPOSITIONS[material]
    return Shoe(shoe_id=f"synthetic_{material}", materials=composition)


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(n_samples: int = 7500, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a synthetic footwear lifecycle dataset for Project S.H.O.E.S.

    Returns a DataFrame with columns:
        category, material, brand_tier,
        condition_score, age_months, original_price,
        resale_market_demand, recyclability_score,
        optimal_path, resale_value

    Label assignment (evaluated in priority order):
        resale  — condition_score > 0.60 AND resale_market_demand > 0.30
                  (~28% of rows; shoes in good condition with market demand)
        recycle — condition_score < 0.20 AND age_months > 30
                  (~15% of rows; truly end-of-life shoes)
                  Note: shoes are notoriously difficult to recycle, so this
                  is intentionally the smallest class.
        repair  — all remaining rows (~57%)
                  Default outcome: the vast majority of shoes have remaining
                  value and can be extended through repair or refurbishment.

    Note on class distribution:
        repair > resale > recycle reflects real-world footwear circularity —
        most shoes benefit from intervention (repair/resale) rather than
        material recovery (recycle), which is technically difficult for
        multi-material footwear construction.
    """
    rng = np.random.default_rng(seed)

    # --- Categorical features ---
    categories  = rng.choice(["sneaker", "boot", "sandal"], size=n_samples)
    materials   = rng.choice(["leather", "synthetic", "mixed"], size=n_samples)
    brand_tiers = rng.choice(["premium", "mid", "fast"], size=n_samples)

    # --- Continuous features ---
    condition_scores      = rng.uniform(0.0, 1.0, size=n_samples)
    age_months            = rng.integers(1, 121, size=n_samples)  # 1–120 inclusive
    original_prices       = rng.uniform(30.0, 500.0, size=n_samples)
    resale_market_demands = rng.uniform(0.0, 1.0, size=n_samples)

    # --- recyclability_score via main.py bridge ---
    recyclability_scores = np.array(
        [recyclability_score(build_shoe_for_material(m), DEFAULT_POLICY) for m in materials]
    )

    # --- Label assignment (np.select: first matching condition wins) ---
    resale_cond = (condition_scores > 0.60) & (resale_market_demands > 0.30)
    recycle_cond = (condition_scores < 0.20) & (age_months > 30)
    optimal_paths = np.select(
        [resale_cond, recycle_cond],
        ["resale", "recycle"],
        default="repair",
    )

    # --- Resale value (computed for all rows; serves as a market reference
    #     even for repair/recycle paths) ---
    brand_multipliers = np.array([BRAND_MULTIPLIERS[b] for b in brand_tiers])
    noise = rng.normal(0, 5.0, size=n_samples)
    resale_values = (
        original_prices * condition_scores * resale_market_demands
        * brand_multipliers + noise
    ).clip(min=0.0)

    return pd.DataFrame({
        "category":             categories,
        "material":             materials,
        "brand_tier":           brand_tiers,
        "condition_score":      condition_scores,
        "age_months":           age_months,
        "original_price":       original_prices,
        "resale_market_demand": resale_market_demands,
        "recyclability_score":  recyclability_scores,
        "optimal_path":         optimal_paths,
        "resale_value":         resale_values,
    })


# ---------------------------------------------------------------------------
# Quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nClass distribution:\n{df['optimal_path'].value_counts()}")
    print(f"\nRecyclability scores by material:\n{df.groupby('material')['recyclability_score'].mean()}")
    print(f"\nResale value stats:\n{df['resale_value'].describe().round(2)}")
