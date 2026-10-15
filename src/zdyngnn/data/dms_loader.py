"""DMS Fitness Score and Replicate Variance Loader for NDM-1.

Parses experimental Deep Mutational Scanning fitness profiles from Chen et al. 2024
(Supplementary Data 3: SD3_NDM_fitness_scores) across Ampicillin, Cefotaxime, and Meropenem.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# The 9 antibiotic concentration conditions evaluated in Chen et al. 2024
DMS_CONDITIONS: List[Dict[str, Union[str, float]]] = [
    {"condition": "32.0_AMP", "drug": "AMP", "conc": 32.0, "score_col": "32.0_AMP_score", "sd_col": "32.0_AMP_score_sd"},
    {"condition": "256.0_AMP", "drug": "AMP", "conc": 256.0, "score_col": "256.0_AMP_score", "sd_col": "256.0_AMP_score_sd"},
    {"condition": "512.0_AMP", "drug": "AMP", "conc": 512.0, "score_col": "512.0_AMP_score", "sd_col": "512.0_AMP_score_sd"},
    {"condition": "2.0_CTX", "drug": "CTX", "conc": 2.0, "score_col": "2.0_CTX_score", "sd_col": "2.0_CTX_score_sd"},
    {"condition": "16.0_CTX", "drug": "CTX", "conc": 16.0, "score_col": "16.0_CTX_score", "sd_col": "16.0_CTX_score_sd"},
    {"condition": "32.0_CTX", "drug": "CTX", "conc": 32.0, "score_col": "32.0_CTX_score", "sd_col": "32.0_CTX_score_sd"},
    {"condition": "0.0625_MEM", "drug": "MEM", "conc": 0.0625, "score_col": "0.0625_MEM_score", "sd_col": "0.0625_MEM_score_sd"},
    {"condition": "0.25_MEM", "drug": "MEM", "conc": 0.25, "score_col": "0.25_MEM_score", "sd_col": "0.25_MEM_score_sd"},
    {"condition": "0.5_MEM", "drug": "MEM", "conc": 0.5, "score_col": "0.5_MEM_score", "sd_col": "0.5_MEM_score_sd"},
]


def load_raw_dms_excel(
    file_path: Union[str, Path] = "DMS Data NDM-1/41467_2024_52614_MOESM6_ESM.xlsx",
    sheet_name: str = "SD3_NDM_fitness_scores",
) -> pd.DataFrame:
    """Load raw DMS Excel table from Chen et al. 2024.

    Args:
        file_path: Path to Excel file.
        sheet_name: Sheet containing NDM-1 fitness scores.

    Returns:
        DataFrame containing all raw columns with numeric types coerced.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DMS data file not found at: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)

    # Ensure position is integer
    df["pos"] = df["pos"].astype(int)

    # Construct single-string mutation identifier e.g. M1A
    df["mutation"] = df["wt"] + df["pos"].astype(str) + df["mut"]

    # Convert score and sd columns to numeric float
    for cond in DMS_CONDITIONS:
        score_col = str(cond["score_col"])
        sd_col = str(cond["sd_col"])
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        if sd_col in df.columns:
            df[sd_col] = pd.to_numeric(df[sd_col], errors="coerce")

    return df


def filter_missense_dms_data(
    df: pd.DataFrame,
    res_start: Optional[int] = 43,
    res_end: Optional[int] = 270,
    exclude_stop_codons: bool = True,
    exclude_synonymous: bool = True,
) -> pd.DataFrame:
    """Filter DMS table to valid missense mutations and target residue range.

    Args:
        df: Raw or loaded DMS DataFrame.
        res_start: Starting residue position (default 43 for 3SPU Chain A).
        res_end: Ending residue position (default 270 for 3SPU Chain A).
        exclude_stop_codons: Whether to drop stop mutations ('*').
        exclude_synonymous: Whether to drop wild-type/synonymous rows.

    Returns:
        Filtered DataFrame with standardized indexing.
    """
    filtered = df.copy()

    if exclude_stop_codons:
        filtered = filtered[filtered["mut"] != "*"]

    if exclude_synonymous:
        # Keep only identity == 'var' and where mut differs from wt
        filtered = filtered[(filtered["identity"] == "var") & (filtered["mut"] != filtered["wt"])]

    if res_start is not None:
        filtered = filtered[filtered["pos"] >= res_start]
    if res_end is not None:
        filtered = filtered[filtered["pos"] <= res_end]

    return filtered.reset_index(drop=True)


def melt_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Transform wide DMS condition columns into a tidy long DataFrame.

    Returns:
        DataFrame with columns: [pos, wt, mut, mutation, group, drug, conc, condition, fitness_score, replicate_sd].
    """
    records = []
    for cond in DMS_CONDITIONS:
        score_col = cond["score_col"]
        sd_col = cond["sd_col"]
        if score_col not in df.columns:
            continue

        subset = df[["pos", "wt", "mut", "mutation", "group", score_col, sd_col]].copy()
        subset = subset.rename(columns={score_col: "fitness_score", sd_col: "replicate_sd"})
        subset["drug"] = cond["drug"]
        subset["conc"] = cond["conc"]
        subset["condition"] = cond["condition"]
        records.append(subset)

    long_df = pd.concat(records, ignore_index=True)
    return long_df


def extract_condition_dataset(
    df: pd.DataFrame,
    condition: str = "256.0_AMP",
) -> pd.DataFrame:
    """Extract a single condition's fitness scores and replicate standard deviations.

    Args:
        df: Filtered or raw DMS DataFrame.
        condition: Condition name matching one of DMS_CONDITIONS (e.g. '256.0_AMP', '16.0_CTX', '0.25_MEM').

    Returns:
        DataFrame with columns: [pos, wt, mut, mutation, group, fitness_score, replicate_sd].
    """
    matched = [c for c in DMS_CONDITIONS if c["condition"] == condition]
    if not matched:
        available = [c["condition"] for c in DMS_CONDITIONS]
        raise ValueError(f"Unknown condition '{condition}'. Available: {available}")

    cond_meta = matched[0]
    score_col = cond_meta["score_col"]
    sd_col = cond_meta["sd_col"]

    result = df[["pos", "wt", "mut", "mutation", "group", score_col, sd_col]].copy()
    result = result.rename(columns={score_col: "fitness_score", sd_col: "replicate_sd"})
    result["condition"] = condition
    result["drug"] = cond_meta["drug"]
    result["conc"] = cond_meta["conc"]
    return result.reset_index(drop=True)
