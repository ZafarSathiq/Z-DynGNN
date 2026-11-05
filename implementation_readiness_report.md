# Implementation Readiness Report (Stage 9)
**Target Document:** `ideation/04. Z-DynGNN Open-Source Chemical Space Docking - Native Navigation of Proprietary .space Files for Resistance-Proof NDM-1 Drug Discovery.md`  
**Active Scope Audited:** `DetailedRange: lines 142–217`  
**Reviewers:** Pragmatic Tech Lead (`egm-implementation-readiness`), Codebase Details Auditor (`egm-codebase-details`), Style Editor (`egm-style`)  
**Standard Applied:** Re-Implementation Standard (`resources/document_standards/re-implementation-standard.md`)  
**Review Preset:** Implementation Plan Only (Stage 9)  

---

## 1. Executive Summary

The implementation plan establishes an open-source Chemical Space Docking (CSD) module under `src/zdyngnn/csd/` to natively parse proprietary `.space` combinatorial libraries and execute multi-conformation conformal resistance screening.

While the architectural vision is clear, the plan currently **fails the Re-Implementation Standard's Zero-Assumption Rule**. A developer attempting to write this code on the first attempt would be forced to guess:
1. The underlying binary/text file format and parsing protocol of `.space` files.
2. The cheminformatics library dependencies (e.g. RDKit) for chemical reaction SMARTS execution.
3. The exact mathematical formulation for docking candidate synthons into Z-DynGNN graph representations.
4. CLI arguments, data schemas, and error-handling edge cases.

---

## 2. Findings by Reviewer Persona

### A. Pragmatic Tech Lead (`egm-implementation-readiness`)

#### Finding 1 [Blocker]: Unspecified `.space` Physical File Format & Extraction Protocol
* **Affected Scope:** `src/zdyngnn/csd/space_parser.py` (lines 150, 161–193)
* **Violation:** Zero-Assumption Rule (Data Schemas & File Formats).
* **Critique:**
  * Proprietary vendor `.space` files (such as Enamine REAL Space) are multi-file container archives or proprietary binary/text databases containing reaction SMARTS tables, synthon building block lists with isotopic exit tags, and reagent availability matrices.
  * The plan provides a Python stub (`class CombinatorialSpace`) with `pass` statements, but fails to document:
    * Is `.space` a tar/zip container, an SQLite database, or a serialized text table?
    * How are attachment points and exit vectors parsed from SMILES (e.g., dummy isotopic labels `[1*]`, `[2*]`, or ChemAxon atom maps)?
    * What is the fallback behavior when an unknown reaction rule or malformed synthon is encountered?
  * An implementer cannot write `space_parser.py` without guessing the file parsing specification.

#### Finding 2 [Blocker]: Undefined Heterogeneous Ligand-Protein Graph Coupling Tensor
* **Affected Scope:** `src/zdyngnn/csd/ensemble_scorer.py` (lines 153) and Tier 2 (lines 74–80)
* **Violation:** Zero-Assumption Rule (Data Structures & APIs).
* **Critique:**
  * In the existing codebase (`src/zdyngnn/models/gnn.py`), Z-DynGNN operates on a homogeneous protein contact network where nodes are 228 amino acid C$\alpha$ atoms and inputs are 34D vectors (`wt_one_hot + dynamics`).
  * The plan states that Z-DynGNN docks candidate synthons directly. However, it fails to specify:
    * How are ligand atoms represented? (Are they appended as new nodes in a heterogeneous graph, or is the ligand encoded as an external 3D perturbation vector on active-site pocket residues?)
    * What are the input tensor shapes for `ensemble_scorer.py`?
    * How are continuous RBF edge distances calculated between ligand atoms and protein C$\alpha$ atoms?
  * Without this mathematical tensor interface, an engineer cannot implement `ensemble_scorer.py`.

#### Finding 3 [Major]: Undeclared Cheminformatics Library Dependencies
* **Affected Scope:** Section 5 (lines 142–158)
* **Violation:** Requirements (Dependencies & Ecosystem Standards).
* **Critique:**
  * Evaluating chemical synthons, SMILES validity, exit vectors, and 3D pharmacophore generation requires an industrial cheminformatics library (e.g., `rdkit`).
  * Neither `rdkit` nor chemical graph dependencies are declared in `pyproject.toml`, `flake.nix`, or Section 5.
  * The plan must specify whether RDKit is required, how it will be packaged in the hermetic Nix Flake toolchain, and whether C++ coordinate embedding is required for 3D conformer generation.

#### Finding 4 [Major]: Missing CLI Interface Contract for `run_chemical_space_search.py`
* **Affected Scope:** Line 157
* **Violation:** Requirements (APIs & Side Effects).
* **Critique:**
  * The script `scripts/run_chemical_space_search.py` is listed without CLI argument signatures, input flags, or output file formats.
  * Must specify:
    * Arguments: `--space-file`, `--receptor-pdb`, `--top-k`, `--output-dir`, `--alpha`.
    * Output: Structured CSV/JSON schema detailing synthon IDs, predicted binding $\Delta G$, and one-sided upper conformal bounds ($U_{\text{upper}}$).

---

### B. Codebase Details Auditor (`egm-codebase-details`)

#### Finding 5 [Major]: Missed Codebase Reuse of `ProteinGraphContext` and Multi-PDB Dynamics
* **Affected Scope:** Section 5 (lines 146–158)
* **Critique:**
  * In Phase II and Step 11, the codebase already established cached multi-conformation dynamics:
    * `data/processed/structure_3spu_ca.npz` (coords, distance matrix)
    * `data/processed/dynamics_4rl0.npz` (Meropenem-bound loop $L8$ clamping dynamics)
    * `ProteinGraphContext.from_processed_caches()` in `src/zdyngnn/models/dataset.py`
  * The implementation plan does not reference these existing modules and caches. It risks reinventing ad-hoc PDB loaders instead of directly reusing `ProteinGraphContext` with `ctx.coords`.

---

### C. Technical Copy Editor (`egm-style`)

#### Finding 6 [Nit]: Unannotated Python Method Stubs
* **Affected Scope:** Lines 174–193
* **Critique:**
  * Code snippets in Section 5 contain empty `pass` statements without docstrings or type signatures for return values.
  * Update method signatures with concrete type hints (`Tuple[Synthon, ...] -> pd.DataFrame`) and explicit parameter documentation.

---

## 3. Actionable Remediation Checklist

To bring `DetailedRange: lines 142–217` to production implementation readiness:

1. **Specify `.space` Parsing Protocol:** Define whether the open-source parser ingests Enamine REAL Synthon CSVs/TSVs + reaction rules, or binary formats, with explicit schema definitions.
2. **Define Heterogeneous Graph Tensor API:** Specify the exact tensor dimensions for ligand-protein interaction layers in `ensemble_scorer.py`:
   $$\mathbf{x}_{\text{lig}} \in \mathbb{R}^{M \times D_{\text{lig}}}, \quad E_{\text{inter}} \in \mathbb{R}^{2 \times K}, \quad \phi(d_{\text{inter}}) \in \mathbb{R}^{K \times 16}$$
3. **Declare Cheminformatics Toolchain:** Add `rdkit` to `pyproject.toml` and `flake.nix` build inputs.
4. **Define CLI Signature for `run_chemical_space_search.py`:** Specify all arguments, flags, default thresholds, and output CSV columns.
5. **Reuse Existing Caches:** Formally wire `ProteinGraphContext` (`3SPU` and `4RL0`) into `ensemble_scorer.py`.
