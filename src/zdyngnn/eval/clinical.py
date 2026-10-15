"""Clinical Validation and Resistance Decision Utility for NDM-1.

Evaluates certified prediction intervals on strictly unexposed hospital variants
from the Beta-Lactamase Database (BLDB) and experimentally validated wet-lab mutants
from Chen et al. (Supplementary Data 5 EC50 assays). Zero training mutations are included.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# The 22 strictly unexposed validation variants (never seen during model training)
# 9 in held-out TEST (completely unseen positions) + 13 in held-out CALIBRATION.
UNEXPOSED_VALIDATION_VARIANTS: List[Dict[str, Union[str, int]]] = [
    # 1. Held-Out TEST (9 variants, zero training or calibration exposure)
    {"mutation": "A74T", "pos": 74, "wt": "A", "mut": "T", "source": "BLDB NDM-10, 38", "phenotype": "Altered loop L3 mobility", "partition": "TEST"},
    {"mutation": "E170K", "pos": 170, "wt": "E", "mut": "K", "source": "BLDB NDM-17", "phenotype": "Loop L8 charge shift", "partition": "TEST"},
    {"mutation": "P185S", "pos": 185, "wt": "P", "mut": "S", "source": "BLDB NDM-58", "phenotype": "Loop destabilization", "partition": "TEST"},
    {"mutation": "G200R", "pos": 200, "wt": "G", "mut": "R", "source": "BLDB NDM-10", "phenotype": "Neutral wild-type-like fitness", "partition": "TEST"},
    {"mutation": "D223Y", "pos": 223, "wt": "D", "mut": "Y", "source": "BLDB NDM-30", "phenotype": "Beta-hairpin active carbapenemase", "partition": "TEST"},
    {"mutation": "K242Q", "pos": 242, "wt": "K", "mut": "Q", "source": "BLDB NDM-56", "phenotype": "Enhanced ampicillin turnover", "partition": "TEST"},
    {"mutation": "H122Q", "pos": 122, "wt": "H", "mut": "Q", "source": "BLDB NDM-77", "phenotype": "Catalytic Zinc Coordinating Ligand 2", "partition": "TEST"},
    {"mutation": "G197Y", "pos": 197, "wt": "G", "mut": "Y", "source": "Chen et al. SD5 EC50 (epi4)", "phenotype": "Validated EC50 resistance maintenance", "partition": "TEST"},
    {"mutation": "I259L", "pos": 259, "wt": "I", "mut": "L", "source": "Chen et al. SD5 EC50 (epi5)", "phenotype": "Validated EC50 high resistance preservation", "partition": "TEST"},

    # 2. Held-Out CALIBRATION (13 variants, held out from model training)
    {"mutation": "M154L", "pos": 154, "wt": "M", "mut": "L", "source": "BLDB NDM-4, 5, 7 (+29)", "phenotype": "Prevalent allosteric stabilization under zinc starvation", "partition": "CALIB"},
    {"mutation": "M154V", "pos": 154, "wt": "M", "mut": "V", "source": "BLDB NDM-11", "phenotype": "Allosteric helix variant", "partition": "CALIB"},
    {"mutation": "G69S", "pos": 69, "wt": "G", "mut": "S", "source": "BLDB NDM-10, 21", "phenotype": "Loop L3 flexibility", "partition": "CALIB"},
    {"mutation": "A55S", "pos": 55, "wt": "A", "mut": "S", "source": "BLDB NDM-25", "phenotype": "Surface loop substitution", "partition": "CALIB"},
    {"mutation": "A99T", "pos": 99, "wt": "A", "mut": "T", "source": "BLDB NDM-55", "phenotype": "Active carbapenemase", "partition": "CALIB"},
    {"mutation": "Q151L", "pos": 151, "wt": "Q", "mut": "L", "source": "BLDB NDM-47", "phenotype": "Alpha-3 helix packing", "partition": "CALIB"},
    {"mutation": "P179R", "pos": 179, "wt": "P", "mut": "R", "source": "BLDB NDM-49", "phenotype": "Loop L8 substitution", "partition": "CALIB"},
    {"mutation": "M248L", "pos": 248, "wt": "M", "mut": "L", "source": "BLDB NDM-22", "phenotype": "Active-site loop packing & high meropenem resistance", "partition": "CALIB"},
    {"mutation": "A252D", "pos": 252, "wt": "A", "mut": "D", "source": "BLDB NDM-51", "phenotype": "Active carbapenemase", "partition": "CALIB"},
    {"mutation": "D267N", "pos": 267, "wt": "D", "mut": "N", "source": "BLDB NDM-43, 44", "phenotype": "C-terminal packing", "partition": "CALIB"},
    {"mutation": "K125R", "pos": 125, "wt": "K", "mut": "R", "source": "Chen et al. SD5 EC50 (epi1)", "phenotype": "Validated EC50 moderate resistance reduction", "partition": "CALIB"},
    {"mutation": "S249G", "pos": 249, "wt": "S", "mut": "G", "source": "Chen et al. SD5 EC50 (epi1)", "phenotype": "Validated EC50 severe deleterious loss of function", "partition": "CALIB"},
    {"mutation": "I246V", "pos": 246, "wt": "I", "mut": "V", "source": "Chen et al. SD5 EC50 (epi5)", "phenotype": "Validated EC50 wild-type-like neutral fitness", "partition": "CALIB"},
]

# Clinical resistance breakpoints based on experimental log2 fitness scores:
DEFAULT_RESISTANT_THRESHOLD = -2.0
DEFAULT_SUSCEPTIBLE_THRESHOLD = -5.0

# Canonical alias for backward compatibility
BLDB_CLINICAL_MUTATIONS = UNEXPOSED_VALIDATION_VARIANTS

@dataclass
class ClinicalDecisionMetrics:
    """Evaluation metrics for certified clinical decision-making."""
    condition: str
    num_samples: int
    num_true_resistant: int
    num_true_susceptible: int
    certified_resistant_count: int
    certified_susceptible_count: int
    ambiguous_triage_count: int
    triage_rate: float
    decisive_rate: float
    containment_rate: float
    false_safety_count: int


def classify_clinical_decision(
    lower: float,
    upper: float,
    mu: float,
    resistant_threshold: float = DEFAULT_RESISTANT_THRESHOLD,
    susceptible_threshold: float = DEFAULT_SUSCEPTIBLE_THRESHOLD,
) -> str:
    """Classify a certified interval into actionable clinical decision states."""
    if lower >= resistant_threshold:
        return "Certified Resistant"
    elif upper < susceptible_threshold:
        return "Certified Susceptible"
    else:
        return "Ambiguous (Triage for Lab)"


def evaluate_clinical_decision_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mu: np.ndarray,
    resistant_threshold: float = DEFAULT_RESISTANT_THRESHOLD,
    susceptible_threshold: float = DEFAULT_SUSCEPTIBLE_THRESHOLD,
    condition: str = "custom",
) -> ClinicalDecisionMetrics:
    """Compute sensitivity, specificity, and triage efficiency of certified bounds."""
    y_arr = np.asarray(y_true, dtype=np.float64)
    l_arr = np.asarray(lower, dtype=np.float64)
    u_arr = np.asarray(upper, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    n = len(y_arr)

    is_true_resistant = y_arr >= resistant_threshold
    is_true_susceptible = y_arr < susceptible_threshold

    calls = [
        classify_clinical_decision(l, u, m, resistant_threshold, susceptible_threshold)
        for l, u, m in zip(l_arr, u_arr, mu_arr)
    ]

    cert_res = sum(1 for c in calls if c == "Certified Resistant")
    cert_susc = sum(1 for c in calls if c == "Certified Susceptible")
    ambiguous = sum(1 for c in calls if c == "Ambiguous (Triage for Lab)")
    false_safety = sum(1 for i, c in enumerate(calls) if is_true_resistant[i] and c == "Certified Susceptible")

    covered = (y_arr >= l_arr) & (y_arr <= u_arr)
    containment = float(np.mean(covered))

    return ClinicalDecisionMetrics(
        condition=condition,
        num_samples=n,
        num_true_resistant=int(np.sum(is_true_resistant)),
        num_true_susceptible=int(np.sum(is_true_susceptible)),
        certified_resistant_count=cert_res,
        certified_susceptible_count=cert_susc,
        ambiguous_triage_count=ambiguous,
        triage_rate=float(ambiguous / max(1, n)),
        decisive_rate=float((cert_res + cert_susc) / max(1, n)),
        containment_rate=containment,
        false_safety_count=false_safety,
    )


def extract_bldb_variant_evaluation(
    predictions_df: pd.DataFrame,
    condition: str = "custom",
    quantile: float = 3.550,
    catalog: Optional[List[Dict[str, Union[str, int]]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Union[float, int, str]]]:
    """Filter predictions DataFrame to the 22 unexposed validation variants and attach certificates."""
    if catalog is None:
        catalog = UNEXPOSED_VALIDATION_VARIANTS
    bldb_dict = {item["mutation"]: item for item in catalog}
    matched_rows = []

    for _, row in predictions_df.iterrows():
        mut_id = str(row["mutation"])
        if mut_id in bldb_dict:
            meta = bldb_dict[mut_id]
            mu = float(row["pred_mu"])
            sigma = float(row["pred_sigma"])
            y = float(row["fitness_score"])
            b = quantile * sigma
            lower = mu - b
            upper = mu + b
            covered = (y >= lower) and (y <= upper)

            call = classify_clinical_decision(lower, upper, mu)

            record = {
                "mutation": mut_id,
                "variants": meta.get("source", meta.get("variants", "Clinical isolate")),
                "phenotype": meta["phenotype"],
                "partition": meta.get("partition", "UNEXPOSED"),
                "pos": int(row["pos"]),
                "wt": str(row["wt"]),
                "mut": str(row["mut"]),
                "fitness_true": y,
                "pred_mu": mu,
                "pred_sigma": sigma,
                "certified_bound": b,
                "lower_bound": lower,
                "upper_bound": upper,
                "covered": covered,
                "clinical_call": call,
            }
            matched_rows.append(record)

    out_df = pd.DataFrame(matched_rows)

    summary = {
        "condition": condition,
        "num_bldb_variants_evaluated": len(out_df),
        "containment_rate": float(out_df["covered"].mean()) if len(out_df) > 0 else 0.0,
        "mean_certified_width": float((out_df["upper_bound"] - out_df["lower_bound"]).mean()) if len(out_df) > 0 else 0.0,
    }

    return out_df, summary
