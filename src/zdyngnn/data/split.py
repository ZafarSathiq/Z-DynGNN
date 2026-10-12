"""Position-Disjoint Splitting for NDM-1 Mutation Datasets.

Enforces strict spatial independence between training, calibration, and test partitions:
    P_train cap P_calib = empty
    P_train cap P_test  = empty
    P_calib cap P_test  = empty

Prevents positional memorization and ensures valid exchangeability for conformal calibration.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class PositionDisjointSplit:
    """Represents a strictly position-disjoint partition of residues and mutations."""
    train_positions: List[int]
    calib_positions: List[int]
    test_positions: List[int]
    train_mutation_indices: List[int] = field(default_factory=list)
    calib_mutation_indices: List[int] = field(default_factory=list)
    test_mutation_indices: List[int] = field(default_factory=list)

    @property
    def num_train_positions(self) -> int:
        return len(self.train_positions)

    @property
    def num_calib_positions(self) -> int:
        return len(self.calib_positions)

    @property
    def num_test_positions(self) -> int:
        return len(self.test_positions)

    @property
    def num_train_mutations(self) -> int:
        return len(self.train_mutation_indices)

    @property
    def num_calib_mutations(self) -> int:
        return len(self.calib_mutation_indices)

    @property
    def num_test_mutations(self) -> int:
        return len(self.test_mutation_indices)

    def to_dict(self) -> Dict[str, Union[List[int], int]]:
        """Serialize split to a dictionary of standard Python types."""
        return {
            "train_positions": [int(p) for p in self.train_positions],
            "calib_positions": [int(p) for p in self.calib_positions],
            "test_positions": [int(p) for p in self.test_positions],
            "train_mutation_indices": [int(i) for i in self.train_mutation_indices],
            "calib_mutation_indices": [int(i) for i in self.calib_mutation_indices],
            "test_mutation_indices": [int(i) for i in self.test_mutation_indices],
            "num_train_positions": self.num_train_positions,
            "num_calib_positions": self.num_calib_positions,
            "num_test_positions": self.num_test_positions,
            "num_train_mutations": self.num_train_mutations,
            "num_calib_mutations": self.num_calib_mutations,
            "num_test_mutations": self.num_test_mutations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionDisjointSplit":
        """Reconstruct PositionDisjointSplit from dictionary."""
        return cls(
            train_positions=[int(p) for p in data["train_positions"]],
            calib_positions=[int(p) for p in data["calib_positions"]],
            test_positions=[int(p) for p in data["test_positions"]],
            train_mutation_indices=[int(i) for i in data.get("train_mutation_indices", [])],
            calib_mutation_indices=[int(i) for i in data.get("calib_mutation_indices", [])],
            test_mutation_indices=[int(i) for i in data.get("test_mutation_indices", [])],
        )


def verify_disjoint_invariants(split: PositionDisjointSplit, df: Optional[pd.DataFrame] = None) -> bool:
    """Verify that positional partitions and mutation indices strictly satisfy disjointness.

    Args:
        split: Partition to inspect.
        df: Optional DataFrame with 'pos' column to check mutation-level integrity.

    Raises:
        AssertionError if any positional or index leakage is detected.

    Returns:
        True if all invariants hold.
    """
    s_train = set(split.train_positions)
    s_calib = set(split.calib_positions)
    s_test = set(split.test_positions)

    # 1. Positional disjointness
    assert s_train.isdisjoint(s_calib), f"Leakage: Train and Calib share positions: {s_train & s_calib}"
    assert s_train.isdisjoint(s_test), f"Leakage: Train and Test share positions: {s_train & s_test}"
    assert s_calib.isdisjoint(s_test), f"Leakage: Calib and Test share positions: {s_calib & s_test}"

    # 2. Mutation index disjointness
    if split.train_mutation_indices and split.calib_mutation_indices and split.test_mutation_indices:
        idx_train = set(split.train_mutation_indices)
        idx_calib = set(split.calib_mutation_indices)
        idx_test = set(split.test_mutation_indices)
        assert idx_train.isdisjoint(idx_calib), "Leakage: Train and Calib share mutation indices."
        assert idx_train.isdisjoint(idx_test), "Leakage: Train and Test share mutation indices."
        assert idx_calib.isdisjoint(idx_test), "Leakage: Calib and Test share mutation indices."

        if df is not None:
            # Verify each mutation index belongs to the declared position set
            df_pos = df["pos"].values
            train_positions_in_df = set(df_pos[split.train_mutation_indices])
            calib_positions_in_df = set(df_pos[split.calib_mutation_indices])
            test_positions_in_df = set(df_pos[split.test_mutation_indices])

            assert train_positions_in_df.issubset(s_train)
            assert calib_positions_in_df.issubset(s_calib)
            assert test_positions_in_df.issubset(s_test)

    return True


def map_split_to_mutation_indices(
    split: PositionDisjointSplit,
    df: pd.DataFrame,
) -> PositionDisjointSplit:
    """Map residue position partitions onto DataFrame row indices.

    Args:
        split: Position-level partition.
        df: DataFrame containing a 'pos' integer column.

    Returns:
        PositionDisjointSplit with populated mutation index lists.
    """
    if "pos" not in df.columns:
        raise KeyError("DataFrame must contain a 'pos' column.")

    pos_series = df["pos"]
    train_set = set(split.train_positions)
    calib_set = set(split.calib_positions)
    test_set = set(split.test_positions)

    train_indices = df.index[pos_series.isin(train_set)].tolist()
    calib_indices = df.index[pos_series.isin(calib_set)].tolist()
    test_indices = df.index[pos_series.isin(test_set)].tolist()

    populated_split = PositionDisjointSplit(
        train_positions=list(split.train_positions),
        calib_positions=list(split.calib_positions),
        test_positions=list(split.test_positions),
        train_mutation_indices=train_indices,
        calib_mutation_indices=calib_indices,
        test_mutation_indices=test_indices,
    )

    verify_disjoint_invariants(populated_split, df)
    return populated_split


def split_positions_disjoint(
    positions: Sequence[int],
    strata: Optional[Sequence[int]] = None,
    train_ratio: float = 0.70,
    calib_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    mutation_df: Optional[pd.DataFrame] = None,
) -> PositionDisjointSplit:
    """Generate a single strictly position-disjoint 70/15/15 partition.

    Args:
        positions: Array or sequence of unique residue positions (e.g. 43..270).
        strata: Optional array of structural stratum bin IDs for stratified sampling.
        train_ratio: Target fraction of positions for training (default 0.70).
        calib_ratio: Target fraction of positions for calibration (default 0.15).
        test_ratio: Target fraction of positions for testing (default 0.15).
        seed: Random seed for reproducibility.
        mutation_df: Optional DataFrame to map row indices.

    Returns:
        PositionDisjointSplit with verified invariants.
    """
    pos_arr = np.array(positions, dtype=np.int32)
    n_total = len(pos_arr)
    rng = np.random.RandomState(seed)

    if strata is not None:
        strata_arr = np.array(strata, dtype=np.int32)
        if len(strata_arr) != n_total:
            raise ValueError(f"Strata length ({len(strata_arr)}) must match positions ({n_total}).")

        unique_strata = np.unique(strata_arr)
        train_pos_list: List[int] = []
        calib_pos_list: List[int] = []
        test_pos_list: List[int] = []

        for s in unique_strata:
            s_mask = strata_arr == s
            s_positions = pos_arr[s_mask].copy()
            rng.shuffle(s_positions)
            n_s = len(s_positions)

            # Balanced allocation across strata
            n_calib = max(1, int(round(n_s * calib_ratio))) if n_s >= 6 else (1 if n_s >= 4 else 0)
            n_test = max(1, int(round(n_s * test_ratio))) if n_s >= 6 else (1 if n_s >= 3 else 0)

            if n_calib + n_test >= n_s:
                n_calib = 1 if n_s >= 3 else 0
                n_test = 1 if n_s >= 2 else 0

            test_subset = s_positions[:n_test]
            calib_subset = s_positions[n_test : n_test + n_calib]
            train_subset = s_positions[n_test + n_calib :]

            train_pos_list.extend([int(p) for p in train_subset])
            calib_pos_list.extend([int(p) for p in calib_subset])
            test_pos_list.extend([int(p) for p in test_subset])

    else:
        # Unstratified random permutation
        perm = pos_arr.copy()
        rng.shuffle(perm)
        n_test = int(round(n_total * test_ratio))
        n_calib = int(round(n_total * calib_ratio))

        test_pos_list = [int(p) for p in perm[:n_test]]
        calib_pos_list = [int(p) for p in perm[n_test : n_test + n_calib]]
        train_pos_list = [int(p) for p in perm[n_test + n_calib :]]

    split = PositionDisjointSplit(
        train_positions=sorted(train_pos_list),
        calib_positions=sorted(calib_pos_list),
        test_positions=sorted(test_pos_list),
    )

    if mutation_df is not None:
        split = map_split_to_mutation_indices(split, mutation_df)
    else:
        verify_disjoint_invariants(split)

    return split


def split_positions_kfold(
    positions: Sequence[int],
    strata: Optional[Sequence[int]] = None,
    k_folds: int = 5,
    calib_fraction_of_train: float = 0.176,  # 15 / 85 approx 0.176 to achieve ~15% calib overall
    seed: int = 42,
    mutation_df: Optional[pd.DataFrame] = None,
) -> List[PositionDisjointSplit]:
    """Generate k-fold cross-validation position-disjoint splits.

    Each position appears in the test partition of exactly one fold.
    In each fold, the remaining k-1 folds are partitioned into train and calibration.

    Args:
        positions: Sequence of unique residue positions.
        strata: Optional stratum IDs for stratified fold assignment.
        k_folds: Number of cross-validation folds (default 5).
        calib_fraction_of_train: Fraction of non-test positions reserved for calibration.
        seed: Random seed.
        mutation_df: Optional mutation DataFrame to map row indices.

    Returns:
        List of k PositionDisjointSplit objects.
    """
    pos_arr = np.array(positions, dtype=np.int32)
    n_total = len(pos_arr)
    rng = np.random.RandomState(seed)

    # Assign fold ID 0..k-1 to each position
    fold_assignments = np.zeros(n_total, dtype=np.int32)

    if strata is not None:
        strata_arr = np.array(strata, dtype=np.int32)
        unique_strata = np.unique(strata_arr)
        for s in unique_strata:
            idx = np.where(strata_arr == s)[0]
            shuffled_idx = idx.copy()
            rng.shuffle(shuffled_idx)
            for i, pos_idx in enumerate(shuffled_idx):
                fold_assignments[pos_idx] = i % k_folds
    else:
        perm = np.arange(n_total)
        rng.shuffle(perm)
        for i, pos_idx in enumerate(perm):
            fold_assignments[pos_idx] = i % k_folds

    splits: List[PositionDisjointSplit] = []

    for fold_id in range(k_folds):
        test_mask = fold_assignments == fold_id
        non_test_mask = ~test_mask

        test_pos = pos_arr[test_mask]
        non_test_pos = pos_arr[non_test_mask].copy()
        rng_fold = np.random.RandomState(seed + fold_id)
        rng_fold.shuffle(non_test_pos)

        n_calib = int(round(len(non_test_pos) * calib_fraction_of_train))
        calib_pos = non_test_pos[:n_calib]
        train_pos = non_test_pos[n_calib:]

        fold_split = PositionDisjointSplit(
            train_positions=sorted([int(p) for p in train_pos]),
            calib_positions=sorted([int(p) for p in calib_pos]),
            test_positions=sorted([int(p) for p in test_pos]),
        )

        if mutation_df is not None:
            fold_split = map_split_to_mutation_indices(fold_split, mutation_df)
        else:
            verify_disjoint_invariants(fold_split)

        splits.append(fold_split)

    return splits
