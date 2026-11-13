#!/usr/bin/env python3
"""Pairwise similarity heatmap from a FASTA alignment."""
import argparse, os, warnings, csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from Bio import SeqIO
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# Clean sans-serif; prefer Arial, fall back gracefully.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"]

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

def coverage_colour(cov, low_rgb, high_rgb):
    """Interpolate low_rgb (0% coverage) -> high_rgb (100% coverage)."""
    if np.isnan(cov):
        return (*low_rgb, 1.0)
    frac = np.clip(cov / 100.0, 0, 1)
    rgb = tuple(l + frac * (h - l) for l, h in zip(low_rgb, high_rgb))
    return (*rgb, 1.0)

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

def write_pairs_csv(path, names, labels, H, G, id_thr, cov_thr):
    """Write pairs where identity AND coverage exceed thresholds,
    sorted by (identity + coverage) descending."""
    n = len(names)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):        # unique pairs, skip diagonal
            if np.isnan(H[i, j]) or np.isnan(G[i, j]):
                continue
            identity = 100 - H[i, j]
            coverage = 100 - G[i, j]
            if identity > id_thr and coverage > cov_thr:
                rows.append((names[i], names[j], labels[i], labels[j],
                             identity, coverage, identity + coverage))
    rows.sort(key=lambda r: r[6], reverse=True)   # highest total first
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["raw_name_1", "raw_name_2", "figure_name_1",
                    "figure_name_2", "identity_pct", "coverage_pct"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3],
                        f"{r[4]:.2f}", f"{r[5]:.2f}"])
    return len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alignment")
    ap.add_argument("metadata")
    ap.add_argument("columns", help="metadata column(s), e.g. 2 or 2,4,5")
    ap.add_argument("-c", "--colour", default="#a83236",
                    help="hex for MAX coverage (default maroon)")
    ap.add_argument("--low-colour", default="#FFFFFF",
                    help="hex for MIN coverage (default white)")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--id-threshold", type=float, default=70.0,
                    help="min %% identity for the pairs CSV (default 70)")
    ap.add_argument("--cov-threshold", type=float, default=70.0,
                    help="min %% coverage for the pairs CSV (default 70)")
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

    cov = 100 - G   # coverage matrix

    fig, ax = plt.subplots(figsize=(1.2 * n + 4, 1.2 * n + 1))
    for i in range(n):
        for j in range(n):
            colour = coverage_colour(cov[i, j], low_rgb, high_rgb)
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                   facecolor=colour,
                                   edgecolor="lightgrey", lw=0.5))
            identity = np.nan if np.isnan(H[i, j]) else 100 - H[i, j]
            if not np.isnan(identity):
                ax.text(j + 0.5, n - 1 - i + 0.5, f"{round(identity)}",
                        ha="center", va="center", color="black",
                        fontweight="bold", fontsize=24)
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5); ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=90, rotation_mode="anchor",
                       ha="right", va="center", fontsize=22)
    ax.set_yticklabels(labels[::-1], fontsize=22)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)

    ttl = title_from_meta(args.metadata)
    fig.suptitle(f"Pairwise Identity between {ttl}s", fontsize=40)

    # Coverage legend (white -> maroon), fixed 0-100
    grad = np.linspace(0, 1, 100)
    axG = fig.add_axes([0.86, 0.35, 0.03, 0.3])
    imgG = np.array([[coverage_colour(v * 100, low_rgb, high_rgb)
                      for v in grad]])
    axG.imshow(imgG.transpose(1, 0, 2), origin="lower", aspect="auto",
               extent=[0, 1, 0, 100])
    axG.set_title("Alignment Coverage (%)", fontsize=25)
    axG.set_xticks([]); axG.set_yticks([0, 50, 100])
    axG.set_yticklabels([0, 50, 100], fontsize=25)

    out = args.output or f"{ttl}_identity_heatmap.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"Saved {out}")

    csv_out = f"{ttl}_pairs.csv"
    n_pairs = write_pairs_csv(csv_out, names, labels, H, G,
                              args.id_threshold, args.cov_threshold)
    print(f"Saved {csv_out} ({n_pairs} pair(s) above thresholds)")

if __name__ == "__main__":
    main()
