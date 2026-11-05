"""Automated unit and integration test for CSD search pipeline."""
import json
from pathlib import Path
import pytest
import pandas as pd

from zdyngnn.csd.pipeline import run_chemical_space_search


def test_run_chemical_space_search_end_to_end(tmp_path: Path):
    """Verify that the end-to-end CSD search executes cleanly and satisfies Specification 5.4."""
    out_dir = tmp_path / "test_csd_out"

    csv_path = run_chemical_space_search(
        top_k=5,
        condition="0.25_MEM",
        output_dir=str(out_dir),
    )

    assert csv_path.exists()
    df = pd.read_csv(csv_path)

    # 1. Exact 11 columns required by Document 04 Section 5.4
    expected_columns = [
        "lead_rank",
        "synthon_assembly_id",
        "reaction_id",
        "smiles",
        "sa_score",
        "pred_delta_g",
        "pred_sigma",
        "allosteric_score",
        "max_u_upper_bldb",
        "escape_resistant_call",
        "vendor_order_code",
    ]
    assert list(df.columns) == expected_columns

    # 2. Row count matches top-k
    assert len(df) == 5

    # 3. Value invariants
    assert (df["lead_rank"] == [1, 2, 3, 4, 5]).all()
    assert (df["pred_sigma"] > 0).all()
    assert (df["sa_score"] >= 1.0).all() and (df["sa_score"] <= 5.0).all()
    assert (df["max_u_upper_bldb"] < -2.0).all()  # Certified resistance-proof
    assert (df["escape_resistant_call"] == "Certified Resistance-Proof").all()

    # Vendor order code format
    for code in df["vendor_order_code"]:
        assert code.startswith("ENAMINE-REAL:")
        assert "+" in code

    # 4. Manifest JSON verification
    manifest_path = out_dir / "csd_search_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert "execution_timestamp" in manifest
    assert "conformal_quantile_upper" in manifest
    assert manifest["target_condition"] == "0.25_MEM"
    assert manifest["resistance_proof_leads_found"] > 0
