"""Empirical Baselines for NDM-1 Mutation Effect Prediction.

Implements mandatory benchmark baselines:
1. Site-Mean Baseline: Predicts the empirical mean fitness of observed positions (or global mean for held-out positions).
2. Ridge Regression Baseline: Linear model combining amino acid substitutions and structural/dynamics features.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Standard 20 amino acid alphabet
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def encode_amino_acid_substitutions(wt_list: List[str], mut_list: List[str]) -> np.ndarray:
    """Encode WT and MUT amino acids as 40-dimensional concatenated one-hot vectors."""
    n = len(wt_list)
    encoding = np.zeros((n, 40), dtype=np.float32)
    for i, (wt, mut) in enumerate(zip(wt_list, mut_list)):
        if wt in AA_TO_IDX:
            encoding[i, AA_TO_IDX[wt]] = 1.0
        if mut in AA_TO_IDX:
            encoding[i, 20 + AA_TO_IDX[mut]] = 1.0
    return encoding


class SiteMeanBaseline:
    """Position-level mean baseline.

    Under position-disjoint splits, unseen test positions fall back to the global
    training mean. This proves whether a deep model has learned structural epistasis
    or merely memorized position-specific average conservation.
    """

    def __init__(self):
        self.site_means: Dict[int, float] = {}
        self.global_mean: float = 0.0
        self.residual_std: float = 1.0

    def fit(self, df: pd.DataFrame, score_col: str = "fitness_score") -> "SiteMeanBaseline":
        """Compute empirical mean fitness per position on training set."""
        self.global_mean = float(df[score_col].mean())
        grouped = df.groupby("pos")[score_col].mean()
        self.site_means = grouped.to_dict()

        # Compute empirical training residual standard deviation
        preds = np.array([self.site_means.get(p, self.global_mean) for p in df["pos"].values])
        residuals = df[score_col].values - preds
        self.residual_std = float(max(1e-4, np.std(residuals)))

        return self

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and scale for each mutation row in df.

        Returns:
            mu: Array of shape (M,), predicted mean fitness.
            sigma: Array of shape (M,), predictive scale.
        """
        mu = np.array([self.site_means.get(p, self.global_mean) for p in df["pos"].values], dtype=np.float32)
        sigma = np.full_like(mu, fill_value=self.residual_std, dtype=np.float32)
        return mu, sigma


class RidgeBaseline:
    """Ridge regression baseline combining amino acid substitution pairs and residue dynamics."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.residual_std: float = 1.0

    def _extract_features(
        self,
        df: pd.DataFrame,
        node_features_map: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        aa_feats = encode_amino_acid_substitutions(df["wt"].tolist(), df["mut"].tolist())
        if node_features_map is not None:
            topo_list = [node_features_map.get(int(p), np.zeros(14, dtype=np.float32)) for p in df["pos"].values]
            topo_feats = np.array(topo_list, dtype=np.float32)
            return np.hstack([aa_feats, topo_feats])
        return aa_feats

    def fit(
        self,
        df: pd.DataFrame,
        node_features_map: Optional[Dict[int, np.ndarray]] = None,
        score_col: str = "fitness_score",
    ) -> "RidgeBaseline":
        """Fit Ridge regression on training set."""
        X = self._extract_features(df, node_features_map)
        y = df[score_col].values.astype(np.float32)
        self.model.fit(X, y)

        preds = self.model.predict(X)
        self.residual_std = float(max(1e-4, np.std(y - preds)))
        return self

    def predict(
        self,
        df: pd.DataFrame,
        node_features_map: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and scale."""
        X = self._extract_features(df, node_features_map)
        mu = self.model.predict(X).astype(np.float32)
        sigma = np.full_like(mu, fill_value=self.residual_std, dtype=np.float32)
        return mu, sigma
