"""Unit and integration tests for Step 2 Data Preprocessing Pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zdyngnn.data.dms_loader import (
    DMS_CONDITIONS,
    extract_condition_dataset,
    filter_missense_dms_data,
    load_raw_dms_excel,
    melt_to_long_format,
)
from zdyngnn.data.noise_ceiling import (
    compute_all_noise_ceilings,
    compute_condition_noise_ceiling,
    compute_normalized_metrics,
)
from zdyngnn.data.pdb_parser import parse_pdb_structure
from zdyngnn.data.pipeline import prepare_and_cache_datasets

PDB_FILE = "DMS Data NDM-1/Strutural data/3SPU.pdb"
DMS_FILE = "DMS Data NDM-1/41467_2024_52614_MOESM6_ESM.xlsx"


def test_pdb_parser_structure():
    """Verify 3SPU Chain A parsing, coordinate shapes, and contact degrees."""
    struct = parse_pdb_structure(
        pdb_path=PDB_FILE,
        chain_id="A",
        res_start=43,
        res_end=270,
        distance_cutoff=8.0,
    )

    assert struct.num_residues == 228
    assert struct.coords.shape == (228, 3)
    assert struct.b_factors.shape == (228,)
    assert struct.dist_matrix.shape == (228, 228)
    assert struct.adj_matrix.shape == (228, 228)
    assert len(struct.sequence) == 228

    # Sequence range integrity
    assert int(struct.residue_numbers.min()) == 43
    assert int(struct.residue_numbers.max()) == 270
    assert len(set(struct.residue_numbers)) == 228

    # Symmetry and zero diagonal checks
    assert np.allclose(struct.dist_matrix, struct.dist_matrix.T)
    assert np.allclose(np.diag(struct.dist_matrix), 0.0)
    assert np.allclose(struct.adj_matrix, struct.adj_matrix.T)
    assert np.all(np.diag(struct.adj_matrix) == 0)

    # Degree distribution empirical audit verification
    # Min = 4, Max = 18, Mean = 10.26 +/- 0.01, Std = 3.10 +/- 0.01
    assert struct.degrees.min() == 4
    assert struct.degrees.max() == 18
    assert np.isclose(struct.degrees.mean(), 10.263, atol=0.01)
    assert np.isclose(struct.degrees.std(), 3.102, atol=0.01)


def test_dms_loader_and_filtering():
    """Verify ingestion, missense filtering, and tidy formatting of DMS tables."""
    raw_df = load_raw_dms_excel(DMS_FILE)
    assert len(raw_df) == 5554

    missense_df = filter_missense_dms_data(
        raw_df,
        res_start=43,
        res_end=270,
        exclude_stop_codons=True,
        exclude_synonymous=True,
    )

    # In 43-270, exactly 4,271 missense variants across 228 residues
    assert len(missense_df) == 4271
    assert missense_df["pos"].nunique() == 228
    assert (missense_df["mut"] != "*").all()
    assert (missense_df["mut"] != missense_df["wt"]).all()
    assert (missense_df["identity"] == "var").all()

    # Condition extraction
    amp256 = extract_condition_dataset(missense_df, "256.0_AMP")
    assert len(amp256) == 4271
    assert "fitness_score" in amp256.columns
    assert "replicate_sd" in amp256.columns

    # Long format reshaping: 4271 * 9 conditions = 38439 rows
    long_df = melt_to_long_format(missense_df)
    assert len(long_df) == 4271 * 9
    assert long_df["condition"].nunique() == 9


def test_noise_ceiling_synthetic_and_empirical():
    """Verify noise ceiling formulas on synthetic data and real DMS conditions."""
    # Synthetic case: true signal var=9.0, noise var=1.0 -> R2_max = 0.90
    np.random.seed(42)
    true_f = np.random.normal(0, 3.0, size=5000)
    noise_sd = np.ones(5000) * 1.0
    obs_y = true_f + np.random.normal(0, noise_sd)

    syn_res = compute_condition_noise_ceiling(obs_y, noise_sd, condition_name="syn")
    assert np.isclose(syn_res.r2_max, 0.90, atol=0.03)
    assert np.isclose(syn_res.r_max, np.sqrt(syn_res.r2_max))
    assert syn_res.snr > 8.0

    # Test normalized metrics calculation
    mock_preds = true_f + np.random.normal(0, 0.5, size=5000)
    metrics = compute_normalized_metrics(obs_y, mock_preds, r2_max=syn_res.r2_max, r_max=syn_res.r_max)
    assert metrics["r2_normalized"] <= 1.05
    assert metrics["pearson_normalized"] <= 1.05

    # Empirical test on filtered DMS data
    raw_df = load_raw_dms_excel(DMS_FILE)
    missense_df = filter_missense_dms_data(raw_df, res_start=43, res_end=270)
    ceilings = compute_all_noise_ceilings(missense_df)

    assert len(ceilings) == len(DMS_CONDITIONS)
    for cond_name, res in ceilings.items():
        assert 0.80 <= res.r2_max <= 0.99
        assert 0.85 <= res.r_max <= 1.00
        assert res.snr > 5.0


def test_data_pipeline_end_to_end_and_caches(tmp_path):
    """Verify full preprocessing pipeline execution, alignment, and cache files."""
    manifest = prepare_and_cache_datasets(
        pdb_path=PDB_FILE,
        dms_path=DMS_FILE,
        output_dir=tmp_path,
        chain_id="A",
        res_start=43,
        res_end=270,
        distance_cutoff=8.0,
    )

    assert manifest["alignment_verified"] is True
    assert manifest["num_resolved_residues"] == 228
    assert manifest["num_missense_variants"] == 4271

    # Verify generated cache files
    npz_path = tmp_path / "structure_3spu_ca.npz"
    assert npz_path.exists()
    npz_data = np.load(npz_path)
    assert npz_data["coords"].shape == (228, 3)
    assert npz_data["adj_matrix"].shape == (228, 228)
    assert len(npz_data["sequence"]) == 228

    wide_path = tmp_path / "dms_missense_wide.csv"
    assert wide_path.exists()
    wide_df = pd.read_csv(wide_path)
    assert len(wide_df) == 4271

    long_path = tmp_path / "dms_missense_long.csv"
    assert long_path.exists()
    long_df = pd.read_csv(long_path)
    assert len(long_df) == 38439

    ceilings_path = tmp_path / "noise_ceilings.csv"
    assert ceilings_path.exists()
    ceilings_df = pd.read_csv(ceilings_path)
    assert len(ceilings_df) == 9
