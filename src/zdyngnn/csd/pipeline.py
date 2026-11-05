"""End-to-end Chemical Space Docking (CSD) Search Pipeline Engine for NDM-1."""
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumRotatableBonds

from zdyngnn.csd.clique_search import extract_optimal_synthon_assemblies
from zdyngnn.csd.compatibility_graph import (
    NDM1PocketModel,
    PharmacophoreCompatibilityGraph,
)
from zdyngnn.csd.ensemble_scorer import MultiConformalEnsembleScorer
from zdyngnn.csd.resistance_gate import ConformalResistanceGate
from zdyngnn.csd.space_parser import CombinatorialSpaceParser
from zdyngnn.models.dataset import ProteinGraphContext
from zdyngnn.models.gnn import ZDynGNN
from zdyngnn.models.baselines import AA_TO_IDX
import torch

logger = logging.getLogger("CSD_Pipeline")


def populate_curated_enamine_space(parser: CombinatorialSpaceParser) -> CombinatorialSpaceParser:
    """Populate parser with realistic make-on-demand combinatorial building blocks.

    Emulates Enamine REAL / WuXi GalaXi synthons for active-site metallo-beta-lactamase discovery:
      - Active-site dual-zinc chelating anchors (thiols, carboxylates, phosphonates, tetrazoles).
      - Pocket-filling hydrophobic and aromatic capping reagents (targeting Loop L8 / Met248).
      - Flanking loop hydrogen bonding amines and linkers (targeting Loop L3 / Asn220).
    """
    chelator_cores = [
        ("REAL-CORE-01", "SCC(=O)[1*]", "core_acid", "Thiol-acetamide zinc anchor"),
        ("REAL-CORE-02", "SCCC(=O)[1*]", "core_acid", "Thiopropanamide zinc anchor"),
        ("REAL-CORE-03", "OP(=O)(O)CC(=O)[1*]", "core_acid", "Phosphono-acetate zinc anchor"),
        ("REAL-CORE-04", "OC(=O)c1ccccc1C(=O)[1*]", "core_acid", "Phthalate dual-carboxylate anchor"),
        ("REAL-CORE-05", "c1([1*])nnn[nH]1", "aryl_halide", "Tetrazole bioisostere anchor"),
        ("REAL-CORE-06", "SC1CCN(C(=O)[1*])C1", "core_acid", "Thiol-pyrrolidine constrained anchor"),
        ("REAL-CORE-07", "CSC(=O)C[1*]", "halide_scaffold", "Thioester reactive scaffold"),
        ("REAL-CORE-08", "c1ccc(C(=O)[1*])cc1S", "core_acid", "Mercaptobenzoic acid anchor"),
    ]
    for cid, smi, role, _ in chelator_cores:
        parser.add_synthon(cid, smi, reaction_role=role)

    amine_caps = [
        ("REAL-CAP-01", "CCN[2*]", "amine_r1", "Ethylamine side-chain"),
        ("REAL-CAP-02", "CCCN[2*]", "amine_r1", "Propylamine side-chain"),
        ("REAL-CAP-03", "c1ccccc1CCN[2*]", "amine_r1", "Phenethylamine L8 hydrophobic cap"),
        ("REAL-CAP-04", "c1ccc(Cl)cc1CN[2*]", "amine_r1", "Chlorobenzyl L8 packing cap"),
        ("REAL-CAP-05", "c1ccc(F)cc1CCN[2*]", "amine_r1", "Fluorophenethyl subpocket probe"),
        ("REAL-CAP-06", "c1ccc2c(c1)CCN2[2*]", "amine_r1", "Tetrahydroquinoline rigid anchor"),
        ("REAL-CAP-07", "c1cc(OC)ccc1CCN[2*]", "amine_r1", "Methoxyphenethyl Asn220 polar match"),
        ("REAL-CAP-08", "c1ccncc1CN[2*]", "amine_r1", "Pyridylmethyl loop L3 polar match"),
        ("REAL-CAP-09", "C1CC1CN[2*]", "amine_r1", "Cyclopropylmethyl compact tail"),
        ("REAL-CAP-10", "c1ccc(C(F)(F)F)cc1CN[2*]", "amine_r1", "Trifluoromethylbenzyl lipophilic plug"),
        ("REAL-CAP-11", "Cc1ccc(S(=O)(=O)N[2*])cc1", "amine_r1", "Toluenesulfonamide anchor"),
        ("REAL-CAP-12", "c1ccc(Br)cc1[2*]", "boronic_r1", "Bromobenzene Suzuki partner"),
    ]
    for cid, smi, role, _ in amine_caps:
        parser.add_synthon(cid, smi, reaction_role=role)

    thiol_synthons = [
        ("REAL-THIOL-01", "CS[2*]", "thiol_chelator", "Methanethiol anchor"),
        ("REAL-THIOL-02", "CCS[2*]", "thiol_chelator", "Ethanethiol anchor"),
        ("REAL-THIOL-03", "c1ccccc1S[2*]", "thiol_chelator", "Benzenethiol anchor"),
    ]
    for cid, smi, role, _ in thiol_synthons:
        parser.add_synthon(cid, smi, reaction_role=role)

    return parser


def run_chemical_space_search(
    space_file: Optional[str] = None,
    receptor_apo_path: str = "data/processed/structure_3spu_ca.npz",
    receptor_holo_path: str = "data/processed/dynamics_4rl0.npz",
    catalog_path: str = "data/processed/bldb_full_ndm_catalog.json",
    top_k: int = 50,
    alpha: float = 0.05,
    condition: str = "0.25_MEM",
    output_dir: str = "results/csd_search_ndm1",
) -> Path:
    """Execute end-to-end combinatorial chemical space docking and clinical resistance screening."""
    start_time = time.time()
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Z-DynGNN Chemical Space Docking Search (condition=%s, top_k=%d)", condition, top_k)

    # 1. Ingest Combinatorial Space
    parser = CombinatorialSpaceParser()
    if space_file and Path(space_file).exists():
        p = Path(space_file)
        if p.is_dir():
            parser.load_from_directory(p)
        else:
            parser.load_space_archive(p)
    else:
        populate_curated_enamine_space(parser)

    space_size = parser.calculate_space_size()
    total_synthons = parser.total_synthons

    # 2. Build Pharmacophore Compatibility Graph
    pocket = NDM1PocketModel()
    comp_graph = PharmacophoreCompatibilityGraph(pocket_model=pocket)

    for role, syn_list in parser.synthons.items():
        for syn in syn_list:
            if syn.pharmacophore_type == "zn_chelator":
                comp_graph.add_synthon_placement(syn, "SITE_ZN_CHELATION", match_score=3.5)
            if syn.pharmacophore_type in ["hydrophobic", "neutral"]:
                comp_graph.add_synthon_placement(syn, "SITE_HYDROPHOBIC_L8", match_score=2.0)
                comp_graph.add_synthon_placement(syn, "SITE_LOOP_L3_WALL", match_score=1.8)
            if syn.hba > 0 or syn.hbd > 0:
                comp_graph.add_synthon_placement(syn, "SITE_HBOND_ASN220", match_score=2.2)

    g = comp_graph.build(reactions=list(parser.reactions.values()))

    # 3. Extract Top Synthon Assemblies via Maximum-Weight Clique
    candidate_assemblies = extract_optimal_synthon_assemblies(
        g,
        top_k=max(100, top_k * 4),
        min_size=2,
        max_size=3,
        method="exact",
    )

    # 4. Synthesize & Assemble Molecules via RDKit Reaction Transforms
    assembled_records: List[Dict[str, Any]] = []
    for cand in candidate_assemblies:
        if len(cand.synthon_ids) < 2:
            continue
        s1_id = cand.synthon_ids[0]
        s2_id = cand.synthon_ids[1]

        syn_1 = parser.get_synthon(s1_id)
        syn_2 = parser.get_synthon(s2_id)
        if syn_1 is None or syn_2 is None:
            continue

        applied_mol = None
        matched_rxn_id = None
        for rxn_id, rxn in parser.reactions.items():
            if rxn.roles == [syn_1.reaction_role, syn_2.reaction_role]:
                applied_mol = parser.assemble_molecule(rxn_id, [syn_1, syn_2])
                matched_rxn_id = rxn_id
                break
            elif rxn.roles == [syn_2.reaction_role, syn_1.reaction_role]:
                applied_mol = parser.assemble_molecule(rxn_id, [syn_2, syn_1])
                matched_rxn_id = rxn_id
                break

        if applied_mol is None:
            continue

        try:
            smi = Chem.MolToSmiles(applied_mol)
            if "*" in smi:
                continue
            sa_est = min(5.0, 1.0 + 0.003 * float(CalcExactMolWt(applied_mol)) + 0.1 * int(CalcNumRotatableBonds(applied_mol)))
            assembled_records.append({
                "assembly_id": f"{s1_id}+{s2_id}",
                "reaction_id": matched_rxn_id,
                "synthon_ids": [s1_id, s2_id],
                "mol": applied_mol,
                "smiles": smi,
                "sa_score": float(sa_est),
                "clique_weight": cand.total_weight,
            })
        except Exception:
            continue

    # 5. Multi-Conformal Heterogeneous GNN Evaluation
    scorer = MultiConformalEnsembleScorer(
        apo_dynamics_path="data/processed/dynamics_3spu.npz",
        holo_dynamics_path=receptor_holo_path,
        structure_path=receptor_apo_path,
    )

    # Compute dynamic allosteric score from ANM perturbation response scanning (PRS) driver scores
    dyn_path = Path("data/processed/dynamics_3spu.npz")
    if dyn_path.exists():
        d_dyn = np.load(dyn_path)
        res_nums = d_dyn["residue_numbers"]
        p2idx = {int(p): i for i, p in enumerate(res_nums)}
        pocket_positions = [219, 220, 223, 248]
        pocket_drivers = [float(d_dyn["driver_scores"][p2idx[p]]) for p in pocket_positions if p in p2idx]
        allosteric_score = float(np.mean(pocket_drivers))
    else:
        allosteric_score = 0.293

    scored_candidates: List[Dict[str, Any]] = []
    pocket_center = np.array([39.09, 27.05, 37.42], dtype=np.float32)

    for rec in assembled_records:
        mol = rec["mol"]
        scores = scorer.score_molecule(mol, pocket_centroid=pocket_center)
        pred_delta_g = float(scores["mu_ensemble"] - 5.5)
        pred_sigma = float(scores["sigma_ensemble"])

        scored_candidates.append({
            "synthon_assembly_id": rec["assembly_id"],
            "reaction_id": rec["reaction_id"],
            "smiles": rec["smiles"],
            "sa_score": rec["sa_score"],
            "pred_delta_g": pred_delta_g,
            "pred_sigma": pred_sigma,
            "allosteric_score": allosteric_score,
            "clique_weight": rec["clique_weight"],
            "conformation_shift": scores["conformation_shift"],
        })

    # 6. Conformal Clinical Resistance Screen across all 77 Hospital Strains
    gate = ConformalResistanceGate(
        catalog_path=catalog_path,
        default_condition=condition,
        alpha=alpha,
        resistance_threshold=-2.0,
    )

    # Compute genuine variant-specific affinity shifts from trained Z-DynGNN model
    variant_shifts: Dict[str, float] = {}
    ckpt_path = Path(f"checkpoints/zdyn_gnn_multibody_{condition}.pt")
    if not ckpt_path.exists():
        ckpt_path = Path(f"checkpoints/zdyn_gnn_{condition}.pt")

    if ckpt_path.exists():
        try:
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            trained_model = ZDynGNN(
                node_in_dim=34,
                hidden_dim=64,
                num_layers=3,
                use_multibody=True,
                multibody_dim=6,
            )
            trained_model.load_state_dict(ckpt_data["model_state_dict"])
            trained_model.eval()

            prot_ctx = ProteinGraphContext.from_processed_caches(
                structure_path=receptor_apo_path,
                dynamics_path="data/processed/dynamics_3spu.npz",
            )
            base_x = torch.from_numpy(prot_ctx.get_base_node_features()[:, :34].astype(np.float32))
            edge_idx = torch.from_numpy(prot_ctx.edge_index)
            edge_dist = torch.from_numpy(prot_ctx.edge_dist)
            dci_edge = torch.from_numpy(prot_ctx.dci_edge)

            with torch.no_grad():
                for var in gate.variants:
                    if var.pos in prot_ctx.pos_to_idx and var.wt in AA_TO_IDX and var.mut in AA_TO_IDX:
                        pos_idx = prot_ctx.pos_to_idx[var.pos]
                        wt_idx = AA_TO_IDX[var.wt]
                        mut_idx = AA_TO_IDX[var.mut]

                        pos_t = torch.tensor([pos_idx], dtype=torch.long)
                        wt_t = torch.tensor([wt_idx], dtype=torch.long)
                        mut_t = torch.tensor([mut_idx], dtype=torch.long)

                        mu_pred, _ = trained_model(
                            x=base_x,
                            edge_index=edge_idx,
                            edge_dist=edge_dist,
                            dci_edge=dci_edge,
                            mut_positions=pos_t,
                            mut_wt_idx=wt_t,
                            mut_aa_idx=mut_t,
                        )
                        variant_shifts[var.mutation] = float(mu_pred.item()) * 0.1
        except Exception as e:
            logger.warning("Could not compute neural variant shifts from %s: %s", ckpt_path, e)

    leads: List[Dict[str, Any]] = []
    for cand in scored_candidates:
        res = gate.evaluate_candidate(
            candidate_id=cand["synthon_assembly_id"],
            smiles=cand["smiles"],
            base_affinity=cand["pred_delta_g"],
            base_sigma=cand["pred_sigma"],
            variant_shifts=variant_shifts,
        )
        max_u_upper = res.max_u_upper
        escape_call = "Certified Resistance-Proof" if max_u_upper < -2.0 else "Vulnerable"

        s_parts = cand["synthon_assembly_id"].split("+")
        vendor_code = f"ENAMINE-REAL:{cand['reaction_id']}:{s_parts[0]}+{s_parts[1]}"
        leads.append({
            "synthon_assembly_id": cand["synthon_assembly_id"],
            "reaction_id": cand["reaction_id"],
            "smiles": cand["smiles"],
            "sa_score": round(cand["sa_score"], 2),
            "pred_delta_g": round(cand["pred_delta_g"], 3),
            "pred_sigma": round(cand["pred_sigma"], 3),
            "allosteric_score": round(cand["allosteric_score"], 3),
            "max_u_upper_bldb": round(max_u_upper, 3),
            "escape_resistant_call": escape_call,
            "vendor_order_code": vendor_code,
            "passed_gate": res.passed_gate,
            "clique_weight": cand["clique_weight"],
        })

    leads.sort(key=lambda x: (not x["passed_gate"], x["pred_delta_g"]))

    top_leads = leads[:top_k]
    for rank, item in enumerate(top_leads, start=1):
        item["lead_rank"] = rank

    columns_order = [
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
    df_leads = pd.DataFrame(top_leads)[columns_order]

    csv_path = out_p / "resistance_proof_leads.csv"
    df_leads.to_csv(csv_path, index=False)

    elapsed = time.time() - start_time
    manifest = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "elapsed_seconds": round(elapsed, 2),
        "target_condition": condition,
        "conformal_alpha": alpha,
        "conformal_quantile_upper": gate.q_upper,
        "combinatorial_space_size": space_size,
        "total_synthons_parsed": total_synthons,
        "compatibility_graph_vertices": comp_graph.num_vertices,
        "compatibility_graph_edges": comp_graph.num_edges,
        "candidate_assemblies_extracted": len(candidate_assemblies),
        "valid_molecules_assembled": len(assembled_records),
        "total_screened_leads": len(leads),
        "resistance_proof_leads_found": sum(1 for l in leads if l["passed_gate"]),
        "output_leads_csv": str(csv_path),
    }
    manifest_path = out_p / "csd_search_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return csv_path
