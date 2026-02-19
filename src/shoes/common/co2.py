"""
CO2 and materials impact calculator for Project S.H.O.E.S.

Provides rule-based estimates for:
    - shoe_co2e_kg        : lifecycle CO2 equivalent for manufacturing the shoe
    - co2_avoided_kg      : CO2 avoided by following the predicted path vs. landfill
    - materials_diverted_kg : physical weight of material diverted from landfill

Anchor value (from paper):
    Sarker et al. (2024), "Sustainable Practices in Footwear Manufacturing"
    "A typical pair of running shoes made of synthetic materials: 14 kg CO2e ± 2.7 kg"
    Study subject: Asics GEL-KAYANO 17, men's US size 9.

All other values are scaled relative to this anchor using category multipliers
and a size heuristic. No ML is used — these are deterministic formulas.
"""

# ---------------------------------------------------------------------------
# CO2e constants
# ---------------------------------------------------------------------------

# Base lifecycle CO2e by material (kg per pair, at sneaker + size 9 baseline).
# Anchor: synthetic = 14.0 (Sarker et al. 2024).
# Leather ~1.5× synthetic: cattle farming methane, tanning chemicals, water pollution
# (paper: leather has the highest environmental impact of all footwear materials).
# Mixed = estimated midpoint.
BASE_CO2E = {
    "leather":   21.0,
    "synthetic": 14.0,
    "mixed":     17.0,
}

# Category multiplier: scales for material volume per shoe.
# Boots use ~35% more material than sneakers; sandals use ~45% less.
CATEGORY_MULT = {
    "boot":    1.35,
    "sneaker": 1.00,  # paper baseline category
    "sandal":  0.55,
}

# Path savings rates: fraction of shoe CO2e avoided vs. landfill/incineration.
# Sources:
#   resale  — avoids new manufacture entirely; small logistics overhead (0.82)
#   repair  — extends lifecycle; paper: second-best circular strategy (0.62)
#   recycle — partial material recovery; paper explicitly notes multi-material
#             footwear is "challenging to recycle" (0.25)
PATH_SAVINGS_RATE = {
    "resale":  0.82,
    "repair":  0.62,
    "recycle": 0.25,
}

# ---------------------------------------------------------------------------
# Materials diverted constants
# ---------------------------------------------------------------------------

# Typical shoe weight by category (kg per pair).
SHOE_WEIGHT_KG = {
    "boot":    0.85,
    "sneaker": 0.60,
    "sandal":  0.30,
}

# Diversion rate for resale/repair paths (shoe kept in use = fully diverted).
RESALE_DIVERSION  = 1.00
REPAIR_DIVERSION  = 0.95

# Recycle diversion rates: pre-computed weighted average of per-material
# recovery rates (rubber: 0.40, polyester: 0.50, leather: 0.20, eva_foam: 0.15)
# weighted by MATERIAL_COMPOSITIONS in data.py.
#   leather  : 0.50×0.20 + 0.30×0.40 + 0.20×0.50 = 0.32
#   synthetic: 0.60×0.50 + 0.25×0.15 + 0.15×0.40 = 0.40
#   mixed    : 0.25×0.20 + 0.35×0.50 + 0.20×0.15 + 0.20×0.40 = 0.34
# Paper: take-back programs recover rubber, leather, and plastic specifically.
RECYCLE_DIVERSION = {
    "leather":   0.32,
    "synthetic": 0.40,
    "mixed":     0.34,
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def shoe_co2e_kg(material: str, category: str, size: int) -> float:
    """
    Estimate lifecycle CO2e (kg) for a shoe based on material, category, and size.

    Anchor: synthetic sneaker at men's US size 9 = 14.0 kg CO2e
    (Sarker et al. 2024, Asics GEL-KAYANO 17 study).

    Size heuristic: ~1.5% material change per size above/below size 9.
    """
    size_mult = 1.0 + (size - 9.0) * 0.015
    return BASE_CO2E[material] * CATEGORY_MULT[category] * size_mult


def co2_avoided_kg(shoe_co2e: float, optimal_path: str) -> float:
    """
    Estimate CO2e (kg) avoided vs. landfilling the shoe and buying a replacement.

    The baseline assumption is: if the shoe is discarded, the owner buys a new
    pair (another full shoe_co2e of emissions). The path savings rate represents
    how much of that replacement manufacture is avoided by choosing each path:
        resale  — displaces new manufacture almost entirely (0.82)
        repair  — extends life, delays next purchase (0.62)
        recycle — partial material recovery offsets new production (0.25)

    Higher is better. Resale avoids the most; recycle the least.
    """
    return shoe_co2e * PATH_SAVINGS_RATE[optimal_path]


def materials_diverted_kg(material: str, category: str, optimal_path: str) -> float:
    """
    Estimate physical weight (kg) of material diverted from landfill.

    For resale/repair: the full (or near-full) shoe weight is diverted.
    For recycle: only the recoverable fraction is diverted, weighted by
    material-specific recovery rates (rubber, polyester, leather, eva_foam).
    """
    shoe_weight = SHOE_WEIGHT_KG[category]

    if optimal_path == "resale":
        return shoe_weight * RESALE_DIVERSION
    elif optimal_path == "repair":
        return shoe_weight * REPAIR_DIVERSION
    else:  # recycle
        return shoe_weight * RECYCLE_DIVERSION[material]


# ---------------------------------------------------------------------------
# Quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CO2 sanity checks ===")
    anchor = shoe_co2e_kg("synthetic", "sneaker", 9)
    print(f"synthetic + sneaker + size 9  : {anchor:.1f} kg CO2e  (paper anchor: 14.0)")
    print(f"synthetic + sneaker + size 12 : {shoe_co2e_kg('synthetic', 'sneaker', 12):.1f} kg CO2e")
    print(f"leather   + boot    + size 9  : {shoe_co2e_kg('leather', 'boot', 9):.1f} kg CO2e")
    print(f"mixed     + sandal  + size 7  : {shoe_co2e_kg('mixed', 'sandal', 7):.1f} kg CO2e")

    print("\n=== CO2 avoided by path (synthetic sneaker size 9) ===")
    for path in ["resale", "repair", "recycle"]:
        print(f"  {path:<8}: {co2_avoided_kg(anchor, path):.1f} kg CO2e avoided")

    print("\n=== Materials diverted by path (synthetic sneaker) ===")
    for path in ["resale", "repair", "recycle"]:
        print(f"  {path:<8}: {materials_diverted_kg('synthetic', 'sneaker', path):.2f} kg")
