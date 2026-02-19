"""
Footwear Recyclability Prototype

Problem:
Given a shoe's material composition and local recycling rules,
predict whether the product is realistically recyclable at end-of-life.

This file intentionally starts simple:
- explicit assumptions
- transparent logic
- easy path to ML later
"""

from dataclasses import dataclass
from typing import Dict


# -----------------------------
# Data definitions
# -----------------------------

@dataclass
class Shoe:
    shoe_id: str
    materials: Dict[str, float]  # material -> proportion (must sum to 1.0)


@dataclass
class RecyclingPolicy:
    city: str
    accepted_materials: Dict[str, bool]  # material -> accepted or not


# -----------------------------
# Example data (synthetic, realistic)
# -----------------------------

SHOES = [
    Shoe(
        shoe_id="shoe_001",
        materials={
            "rubber": 0.40,
            "polyester": 0.30,
            "eva_foam": 0.30,
        },
    ),
    Shoe(
        shoe_id="shoe_002",
        materials={
            "leather": 0.50,
            "rubber": 0.30,
            "polyester": 0.20,
        },
    ),
]


BOSTON_RECYCLING = RecyclingPolicy(
    city="Boston",
    accepted_materials={
        "rubber": False,
        "polyester": True,
        "eva_foam": False,
        "leather": False,
    },
)


# -----------------------------
# Core logic (baseline rule-based)
# -----------------------------

def recyclability_score(shoe: Shoe, policy: RecyclingPolicy) -> float:
    """
    Returns the fraction of the shoe (by material weight)
    that is accepted by the local recycling system.
    """
    score = 0.0
    for material, fraction in shoe.materials.items():
        if policy.accepted_materials.get(material, False):
            score += fraction
    return score


def classify_recyclability(score: float, threshold: float = 0.5) -> str:
    """
    Simple classification:
    - recyclable
    - not recyclable

    This will later be replaced by an ML model.
    """
    return "recyclable" if score >= threshold else "not_recyclable"


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    for shoe in SHOES:
        score = recyclability_score(shoe, BOSTON_RECYCLING)
        label = classify_recyclability(score)

        print(
            f"Shoe {shoe.shoe_id} | "
            f"City: {BOSTON_RECYCLING.city} | "
            f"Recyclability score: {score:.2f} | "
            f"Label: {label}"
        )
