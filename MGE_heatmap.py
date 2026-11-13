#!/usr/bin/env python3
"""Pairwise similarity heatmap from a FASTA alignment."""
import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from Bio import SeqIO
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

GAP = set("-.")

def read_alignment(path):
    recs = list(SeqIO.parse(path, "fasta"))
    names = [r.id for r in recs]
    seqs = []
    for r in recs:
        s = str(r.seq).upper()
        n_count = s.count("N")
        if n_count:
            total = len(s)
            pct = 100 * n_count / total if total else 0
            warnings.warn(
                f"Sequence '{r.id}': converted {n_count} N(s) to gaps "
                f"({pct:.2f}% of its {total} bp).")
            s = s.replace("N", "-")
        seqs.append(s)
    if len({len(s) for s in seqs}) != 1:
        raise ValueError("Sequences are not all the same length (not aligned).")
    return names, seqs

def read_metadata(path):
    """Read CSV or TSV; first column is the key matching FASTA headers."""
    sep = "\t" if path.lower().endswith((".tsv", ".tab", ".txt")) else ","
    df = pd.read_csv(path, sep=sep, dtype=str)
    df = df.set_index(df.columns[0])
    return df

def pairwise_stats(s1, s2):
    """Return (H_percent, G_percent) for a sequence pair."""
    hamming = onegap = kept = 0
    for a, b in zip(s1, s2):
        ag, bg = a in GAP, b in GAP
        if ag and bg:
            continue                # drop double gaps
        kept += 1                   # contributes to L
        if ag != bg:
            onegap += 1             # exactly one gap
        elif a != b:
            hamming += 1
    if kept == 0:
        return np.nan, np.nan
    G = 100 * onegap / kept
    ungapped = kept - onegap
    H = 100 * hamming / ungapped if ungapped else np.nan
    return H, G

def resolve_labels(names, df, cols_arg):
    """cols_arg like '2' or '2,4,5' -> "val2 (val4, val5)".
    Column numbers are 1-based over the ORIGINAL file (key col = 1).
    Up to three bracketed values are supported."""
    cols = []
    for c in str(cols_arg).split(","):
        c = c.strip()
        if c.isdigit():
            cols.append(df.columns[int(c) - 2])   # key col removed -> -2
        else:
            cols.append(c)
    if len(cols) > 4:
        warnings.warn("More than 3 bracketed values requested; using first 3.")
        cols = cols[:4]
    labels = []
    for n in names:
        if n not in df.index:
            labels.append(n)
            continue
        vals = [str(df.at[n, c]) for c in cols]
        labels.append(vals[0] if len(vals) == 1
                      else f"{vals[0]} ({', '.join(vals[1:])})")
    return labels

def cell_colour(identity, low_rgb, high_rgb, alpha):
    """Interpolate low_rgb (0% identity) -> high_rgb (100% identity)."""
    frac = 0.0 if np.isnan(identity) else np.clip(identity / 100.0, 0, 1)
    rgb = tuple(l + frac * (h - l) for l, h in zip(low_rgb, high_rgb))
    return (*rgb, alpha)

def cluster_order(H):
    """Order indices by hierarchical clustering on the H% matrix."""
    n = H.shape[0]
    if n < 3:
        return list(range(n))
    D = H.copy()
    if np.any(~np.isnan(D)):
        D[np.isnan(D)] = np.nanmax(D)
    else:
        D[:] = 0.0
    D = (D + D.T) / 2          # enforce symmetry
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return list(leaves_list(Z))

def title_from_meta(path):
    base = os.path.splitext(os.path.basename(path))[0]
    for suf in ("_metadata", "_meta"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alignment")
    ap.add_argument("metadata")
    ap.add_argument("columns", help="metadata column(s), e.g. 2 or 2,4,5")
    ap.add_argument("-c", "--colour", default="#FFA500",
                    help="hex for MAX identity (default orange)")
    ap.add_argument("--low-colour", default="#000000",
                    help="hex for MIN identity (default black)")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--min-alpha", type=float, default=0.0,
                    help="floor for opacity of very gappy cells")
    args = ap.parse_args()

    names, seqs = read_alignment(args.alignment)
    df = read_metadata(args.metadata)
    low_rgb = to_rgb(args.low_colour)
    high_rgb = to_rgb(args.colour)
    n = len(names)

    H = np.full((n, n), np.nan)
    G = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            H[i, j], G[i, j] = pairwise_stats(seqs[i], seqs[j])

    order = cluster_order(H)
    H, G = H[np.ix_(order, order)], G[np.ix_(order, order)]
    names = [names[k] for k in order]
    labels = resolve_labels(names, df, args.columns)

    # coverage = 100 - G; scale opacity to max off-diagonal coverage
    cov = 100 - G
    offdiag = ~np.eye(n, dtype=bool)
    valid = offdiag & ~np.isnan(cov)
    max_cov = np.nanmax(cov[valid]) if np.any(valid) else 100.0
    if max_cov <= 0:
        max_cov = 100.0

    fig, ax = plt.subplots(figsize=(1.2 * n + 4, 1.2 * n + 1))
    for i in range(n):
        for j in range(n):
            if np.isnan(cov[i, j]):
                a = 1.0
            else:
                a = np.clip(cov[i, j] / max_cov, 0, 1)
            a = max(a, args.min_alpha)
            identity = np.nan if np.isnan(H[i, j]) else 100 - H[i, j]
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                   facecolor=cell_colour(identity, low_rgb,
                                                         high_rgb, a),
                                   edgecolor="lightgrey", lw=0.5))
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5); ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=90, rotation_mode="anchor",
                       ha="right", va="center")
    ax.set_yticklabels(labels[::-1])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)

    ttl = title_from_meta(args.metadata)
    fig.suptitle(f"Pairwise Identity between {ttl}s", fontsize=28)

    grad = np.linspace(0, 1, 100)
    # Identity legend: low->high colour, fully opaque
    axH = fig.add_axes([0.86, 0.55, 0.03, 0.3])
    imgH = np.array([[cell_colour(v * 100, low_rgb, high_rgb, 1.0)
                      for v in grad]])
    axH.imshow(imgH.transpose(1, 0, 2), origin="lower", aspect="auto",
               extent=[0, 1, 0, 100])
    axH.set_title("Nucleotide Identity (%)", fontsize=11)
    axH.set_xticks([]); axH.set_yticks([0, 50, 100])

    # Coverage legend: opacity gradient at max identity colour
    axG = fig.add_axes([0.86, 0.15, 0.03, 0.3])
    imgG = np.array([[cell_colour(100, low_rgb, high_rgb, v) for v in grad]])
    axG.imshow(imgG.transpose(1, 0, 2), origin="lower", aspect="auto",
               extent=[0, 1, 0, 100])
    axG.set_title("Alignment Coverage (%)", fontsize=11)
    axG.set_xticks([]); axG.set_yticks([0, 50, 100])
    axG.set_yticklabels(["0", "50", f"{a:.2f}:100"])

    out = args.output or f"{ttl}_identity_heatmap.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
