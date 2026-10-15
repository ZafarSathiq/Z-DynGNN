#!/usr/bin/env python3
"""Fetch and catalog all NDM clinical variants (NDM-1 through NDM-77) from BLDB."""
import json
import time
import urllib.request
from pathlib import Path
import pandas as pd

OUTPUT_PATH = Path("data/processed/bldb_full_ndm_catalog.json")


def fetch_seq(v: int, max_retries: int = 3):
    url = f"http://www.bldb.eu/seq_prot/B1-NDM-{v}-prot.fasta"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5.0) as r:
                lines = r.read().decode("utf-8").splitlines()
                seq = "".join([l.strip() for l in lines if not l.startswith(">")])
                if len(seq) == 270:
                    return seq
        except Exception:
            time.sleep(0.3)
    return None


def main():
    print("[*] Fetching NDM-1 wild-type sequence from BLDB...")
    wt_seq = fetch_seq(1)
    if not wt_seq or len(wt_seq) != 270:
        raise RuntimeError("Failed to fetch NDM-1 WT sequence from BLDB.")

    print(f"    WT NDM-1 length: {len(wt_seq)} residues.")

    all_variants = {}
    mutations_map = {}

    print("[*] Fetching NDM-2 through NDM-77 from BLDB...")
    successful_fetches = 0
    for v in range(2, 78):
        s = fetch_seq(v)
        if s and len(s) == len(wt_seq):
            successful_fetches += 1
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

    print(f"[+] Successfully fetched {successful_fetches} NDM variant sequences from BLDB.")
    print(f"[+] Cataloged {len(all_variants)} NDM strains.")
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

    print(f"[+] Cached full BLDB catalog to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
