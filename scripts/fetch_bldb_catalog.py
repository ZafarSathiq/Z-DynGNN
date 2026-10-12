#!/usr/bin/env python3
"""Fetch and catalog all NDM clinical variants (NDM-1 through NDM-69) from BLDB."""
import json
import urllib.request
from pathlib import Path
import pandas as pd

OUTPUT_PATH = Path("data/processed/bldb_ndm_variants_catalog.json")


def fetch_seq(v: int):
    url = f"http://www.bldb.eu/seq_prot/B1-NDM-{v}-prot.fasta"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3.0) as r:
            lines = r.read().decode("utf-8").splitlines()
            return "".join([l.strip() for l in lines if not l.startswith(">")])
    except Exception:
        return None


def main():
    print("[*] Fetching NDM-1 wild-type sequence from BLDB...")
    wt_seq = fetch_seq(1)
    if not wt_seq or len(wt_seq) != 270:
        raise RuntimeError("Failed to fetch NDM-1 WT sequence from BLDB.")

    print(f"    WT NDM-1 length: {len(wt_seq)} residues.")

    all_variants = {}
    mutations_map = {}

    print("[*] Fetching NDM-2 through NDM-69 from BLDB...")
    for v in range(2, 70):
        s = fetch_seq(v)
        if s and len(s) == len(wt_seq):
            diffs = []
            for i in range(len(s)):
                if s[i] != wt_seq[i]:
                    pos = i + 1
                    mut_str = f"{wt_seq[i]}{pos}{s[i]}"
                    diffs.append({"pos": pos, "mutation": mut_str, "wt": wt_seq[i], "mut": s[i]})

                    if 43 <= pos <= 270:
                        if mut_str not in mutations_map:
                            mutations_map[mut_str] = {
                                "mutation": mut_str,
                                "pos": pos,
                                "wt": wt_seq[i],
                                "mut": s[i],
                                "variants": [f"NDM-{v}"],
                            }
                        else:
                            mutations_map[mut_str]["variants"].append(f"NDM-{v}")

            all_variants[f"NDM-{v}"] = diffs

    print(f"[+] Successfully cataloged {len(all_variants)} NDM variants from BLDB.")
    print(f"[+] Found {len(mutations_map)} unique single substitutions in mature region (43-270).")

    catalog = {
        "num_variants_cataloged": len(all_variants),
        "unique_substitutions_in_43_270": len(mutations_map),
        "variants": all_variants,
        "mutations": mutations_map,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"[+] Cached BLDB catalog to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
