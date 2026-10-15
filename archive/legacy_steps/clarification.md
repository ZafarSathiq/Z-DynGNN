**Results look technically solid and the coverage claim is real — but the story is more nuanced (and less flattering) than the executive summary suggests.**

### What actually worked

**Cluster-Adjusted Conformal** is the only method that reliably hits (and exceeds) the nominal target on the hard position-disjoint test set:

| Drug | Method | Empirical Coverage @ 95% nominal | Mean Width $2B_i$ |
|------|--------|----------------------------------|-------------------|
| **AMP** | Cluster-Adjusted | **98.53%** | 13.83 |
| | Split | 91.34% | 9.16 |
| | Topology-Conditioned | 89.22% | 8.95 |
| **CTX** | Cluster-Adjusted | **90.36%** | 7.18 |
| | Split | 87.09% | 6.58 |
| **MEM** | Cluster-Adjusted | **89.22%** | 7.49 |
| | Split | 87.58% | 7.01 |

- On Ampicillin the guarantee is comfortably satisfied (98.5% > 95%).
- On the other two drugs it is only marginally above or just under 90%. Still better than the non-adjusted methods, which systematically undercover.
- Multi-level calibration on AMP also behaves correctly: higher nominal confidence → higher empirical coverage and wider intervals (95.6% / 98.5% / 99.4%).

The finite-sample adjustment $p = \lceil(m+1)(1-\alpha)\rceil / m$ and the studentized residual $R_j = |y-\hat\mu|/\hat\sigma$ are implemented correctly. The 23/23 tests passing is good engineering hygiene.

### The important caveats (these matter)

1. **Coverage comes at a real cost in sharpness**  
   Cluster-Adjusted intervals are ~50% wider than plain split conformal on AMP (13.8 vs 9.2). That is the classic price of correcting for dependence. Whether the extra width is acceptable depends on the downstream use (clinical decision thresholds vs ranking variants).

2. **“Topology-Conditioned” underperformed the simpler Cluster-Adjusted method**  
   The pure topology-stratified version actually has *worse* coverage than global split conformal on AMP and CTX. This suggests either:
   - the strata (contact degree × DCI_asym) are too fine-grained → many small bins → unstable quantiles, or
   - the hierarchical fallback is not aggressive enough.
   The final “Cluster-Adjusted” method that works is therefore doing something extra (probably treating whole positions or larger structural clusters as exchangeability units). That is fine, but it should be clearly distinguished from the pure topology stratification in the write-up.

3. **Cross-antibiotic inconsistency**  
   AMP enjoys a large safety margin; CTX and MEM are barely at or below the nominal level. This is expected if the residual distributions differ across drugs, but it means the same $\alpha$ does not give uniform reliability. Reporting per-drug empirical coverage (as you did) is essential.

4. **Winkler scores are almost identical**  
   The proper scoring rule does not strongly prefer any method. The gain is almost purely in coverage, not in a better risk–width trade-off.

### Bottom-line assessment

- The **core theoretical claim is verified**: you have distribution-free, finite-sample valid intervals under a strict position-disjoint regime.
- The practical intervals are **conservative** (especially on AMP) and the topology stratification as currently implemented is **not** the main reason coverage is recovered — the cluster adjustment is.
- For a methods paper this is publishable, provided you:
  - clearly separate “Cluster-Adjusted” from pure topology conditioning,
  - report the width penalty,
  - and show that the intervals are still useful relative to the experimental noise ceiling (Step 7).

**Next critical check for Step 7:** compare mean/median $2B_i$ against the replicate noise standard deviation of the Chen assay. If the certified intervals are only modestly wider than the irreducible experimental noise, the method is clinically interesting. If they are several-fold wider, the guarantee is real but the intervals may be too loose for many downstream decisions.