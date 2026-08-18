import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========= Configuration =========

TFBS_DIR = "Sequence/2.0 data"
TFBS_PATTERN = "*_tfbs_scan.csv"

WINDOW_SIZE = 100
WINDOW_STEP = 20

TOP_N_HITS = 100

OUT_FIG_DIR = "Sequence/tfbs_analysis_figs"
os.makedirs(OUT_FIG_DIR, exist_ok=True)


# ========= TF family assignment =========
def assign_family(tf: str) -> str:
    t = tf.upper()

    # --- Specific TFs of interest, matched first ---
    if "YY" in t: return "YY1 Family"             # YY1, YY2 (zinc finger)
    if "NFI" in t: return "NFI"                   # NFIA/B/C/X (CCAAT-binding)
    if "NFAT" in t: return "NFAT"                 # NFATC1-4 (immune)
    if "GMEB" in t: return "GMEB"                 # GMEB1/2 (zinc finger)
    if "ZGLP" in t: return "GATA-like (ZGLP)"     # GATA-like zinc finger

    # --- GC-rich factors and named zinc fingers ---
    if "KLF" in t: return "KLF"
    if t.startswith("SP") and len(t) <= 4: return "SP"
    if "MAZ" in t or "VEZF" in t: return "SP-like"
    if "HIC" in t: return "HIC (ZF)"
    if "PLAG" in t: return "PLAG (ZF)"
    if "CTCF" in t: return "CTCF"                 # insulator
    if "ZEB" in t or "SNAI" in t or "SLUG" in t: return "EMT-TF (ZF)"
    if "EGR" in t or "WT1" in t: return "EGR/WT1"

    # Generic zinc fingers. Checked after the named ones above so that
    # specific families (e.g. GATA) are not swallowed by this catch-all.
    zf_keywords = ["ZNF", "ZBTB", "ZSCAN", "ZFX", "ZIC", "ZFP", "ZKSCAN", "ZIM", "ZBED"]
    if any(k in t for k in zf_keywords): return "Zinc Finger (General)"

    # --- Large structural families ---

    # ETS: ELF, ELK, ETS, ERG, FLI, GABPA, SPI, ETV, FEV
    ets_keywords = ["ETS", "ELK", "ELF", "ERG", "FLI", "GABPA", "SPI", "ETV", "FEV"]
    if any(k in t for k in ets_keywords): return "ETS"

    # Homeobox: HOX, PAX, SOX, POU, DLX, LHX, NKX, SIX, MEIS, PBX, OTX
    homeo_keywords = ["HOX", "PAX", "SOX", "POU", "DLX", "LHX", "NKX", "SIX",
                      "MEIS", "PBX", "OTX", "NANOG", "OCT4", "CDX", "MIX", "ISL", "GBX"]
    if any(k in t for k in homeo_keywords): return "Homeobox"

    # bHLH: MYC, MAX, USF, TCF, TWIST, NEURO, ASCL, HIF, HAND, HEY
    bhlh_keywords = ["MYC", "MAX", "USF", "TCF", "TWIST", "NEURO", "ASCL",
                     "HIF", "HAND", "HEY", "BHLH", "NHLH", "OLIG"]
    if any(k in t for k in bhlh_keywords): return "bHLH"

    # bZIP: CREB/ATF, AP-1 (FOS/JUN), C/EBP, MAF, NFE2, BACH
    if any(k in t for k in ["CREB", "ATF"]): return "CREB/ATF"
    if any(k in t for k in ["FOS", "JUN", "AP1"]): return "AP-1"
    if "CEBP" in t: return "C/EBP"
    if any(k in t for k in ["MAF", "NFE2", "BACH", "DBP", "HLF", "TEF"]): return "bZIP (Other)"

    if "FOX" in t: return "Forkhead (FOX)"

    if "GATA" in t: return "GATA"

    # Nuclear receptors: NR, ESR, AR, RAR, RXR, PPAR, VDR, GR, PR, ROR, HNF4, COUP
    nr_keywords = ["NR", "ESR", "AR", "RAR", "RXR", "PPAR", "VDR", "GR", "PR", "ROR", "HNF4", "COUP"]
    if any(k in t for k in nr_keywords) or t.startswith("NR"): return "Nuclear Receptor"

    # --- Other functional families ---

    if t.startswith("E2F") or "TFDP" in t: return "E2F"

    if "TFAP2" in t or "AP-2" in t: return "AP-2"

    if "STAT" in t: return "STAT"
    if "IRF" in t: return "IRF"
    if "REL" in t or "NFKB" in t: return "NF-kB"

    if "RUNX" in t: return "Runt (RUNX)"

    if "TBX" in t or "TBR" in t or "EOMES" in t: return "T-box"

    # HMG box other than SOX
    if "LEF" in t or "TCF" in t or "HMG" in t: return "HMG"

    if "TP53" in t or "P53" in t or "P63" in t or "P73" in t: return "p53 Family"

    if "TBP" in t or "GTF" in t: return "Basal Machinery"

    if "SRF" in t or "MEF2" in t: return "MADS box"

    if "TEAD" in t: return "TEAD"

    return "Other"


# ========= Per-promoter analysis =========

def analyze_single_tfbs(csv_path: str):
    promoter_name = os.path.basename(csv_path).replace("_tfbs_scan.csv", "")
    print(f"\n=== Analyzing promoter: {promoter_name} ===")

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  No hits, skip.")
        return None

    score_col = "EffectiveRelScore" if "EffectiveRelScore" in df.columns else "RelScore"

    df = df.sort_values(score_col, ascending=False).head(TOP_N_HITS).copy()
    print(f"  Using top {len(df)} hits by {score_col}")

    df["Family"] = df["TF"].apply(assign_family)

    tf_counts = df["TF"].value_counts()
    fam_counts = df["Family"].value_counts()

    print("  TF counts (top 30):")
    print(tf_counts.head(30))
    print("  Family counts:")
    print(fam_counts)

    # Sliding-window site density along the promoter
    prom_len = df["End"].max()
    windows = []
    for start in range(1, prom_len + 1, WINDOW_STEP):
        end = start + WINDOW_SIZE - 1
        if end > prom_len:
            end = prom_len
        mask = (df["Start"] <= end) & (df["End"] >= start)
        count = mask.sum()
        windows.append({"WindowStart": start, "WindowEnd": end, "Count": count})
        if end == prom_len:
            break
    dens_df = pd.DataFrame(windows)

    # Same sliding window, split by TF family
    fam_list = sorted(df["Family"].unique())
    fam_dens_rows = []
    for fam in fam_list:
        df_fam = df[df["Family"] == fam]
        for start in range(1, prom_len + 1, WINDOW_STEP):
            end = start + WINDOW_SIZE - 1
            if end > prom_len:
                end = prom_len
            mask = (df_fam["Start"] <= end) & (df_fam["End"] >= start)
            count = mask.sum()
            fam_dens_rows.append({
                "Family": fam,
                "WindowStart": start,
                "WindowEnd": end,
                "Count": count,
            })
            if end == prom_len:
                break
    fam_dens_df = pd.DataFrame(fam_dens_rows)

    # One-dimensional positional clustering: sites within gap_thresh bp
    # of the previous site are merged into the same cluster.
    gap_thresh = 50
    df_sorted = df.sort_values("Start")
    clusters = []
    current_cluster = []
    last_pos = None

    for _, row in df_sorted.iterrows():
        pos = row["Start"]
        if last_pos is None or pos - last_pos <= gap_thresh:
            current_cluster.append(row)
        else:
            if current_cluster:
                clusters.append(pd.DataFrame(current_cluster))
            current_cluster = [row]
        last_pos = pos
    if current_cluster:
        clusters.append(pd.DataFrame(current_cluster))

    cluster_summary = []
    for idx, cdf in enumerate(clusters, start=1):
        center = (cdf["Start"].min() + cdf["End"].max()) / 2.0
        span = cdf["End"].max() - cdf["Start"].min() + 1
        cluster_summary.append({
            "ClusterID": idx,
            "CenterPos": center,
            "Span": span,
            "NumSites": len(cdf),
        })
    cluster_df = pd.DataFrame(cluster_summary)
    print("  Clusters found:", len(cluster_df))
    if not cluster_df.empty:
        print(cluster_df[["ClusterID", "CenterPos", "Span", "NumSites"]])

    # ===== Per-promoter figures =====

    # Family counts
    plt.figure(figsize=(8, 4))
    fam_counts.sort_values(ascending=False).plot(kind="bar", color="steelblue")
    plt.ylabel("Number of TFBS")
    plt.title(f"{promoter_name}: TF family counts (top {TOP_N_HITS} hits)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, f"{promoter_name}_family_counts.png"), dpi=300)
    plt.close()

    # Site density along the promoter
    plt.figure(figsize=(10, 4))
    centers = (dens_df["WindowStart"] + dens_df["WindowEnd"]) / 2.0
    plt.plot(centers, dens_df["Count"], marker="o", linestyle="-")
    plt.xlabel("Position on promoter (bp)")
    plt.ylabel(f"TFBS count in {WINDOW_SIZE} bp window")
    plt.title(f"{promoter_name}: TFBS density (top {TOP_N_HITS} hits)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, f"{promoter_name}_density.png"), dpi=300)
    plt.close()

    # Density by family, as a heatmap
    fam_dens_df["WindowCenter"] = (fam_dens_df["WindowStart"] + fam_dens_df["WindowEnd"]) / 2.0
    pivot = fam_dens_df.pivot_table(
        index="Family", columns="WindowCenter", values="Count", fill_value=0
    )

    plt.figure(figsize=(max(8, len(pivot.columns) * 0.3), max(4, len(pivot.index) * 0.3)))
    sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "TFBS count"})
    plt.xlabel("Position (window center, bp)")
    plt.ylabel("TF family")
    plt.title(f"{promoter_name}: TFBS density by family (top {TOP_N_HITS} hits)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, f"{promoter_name}_family_density_heatmap.png"), dpi=300)
    plt.close()

    # Cluster schematic
    if not cluster_df.empty:
        plt.figure(figsize=(10, 2 + 0.3 * len(cluster_df)))
        for _, row in cluster_df.iterrows():
            y = row["ClusterID"]
            x1 = row["CenterPos"] - row["Span"] / 2
            x2 = row["CenterPos"] + row["Span"] / 2
            plt.plot([x1, x2], [y, y], color="black", linewidth=4)
            plt.text(
                x2 + 5, y,
                f"ID {int(row['ClusterID'])}, n={int(row['NumSites'])}",
                va="center",
                fontsize=8
            )
        plt.yticks([])
        plt.xlabel("Position on promoter (bp)")
        plt.title(f"{promoter_name}: TFBS clusters (gap<={gap_thresh}bp)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_FIG_DIR, f"{promoter_name}_clusters.png"), dpi=300)
        plt.close()

    # Export the annotated hit table and the cluster summary
    df.to_csv(os.path.join(OUT_FIG_DIR, f"{promoter_name}_top{TOP_N_HITS}_with_family.csv"), index=False)
    cluster_df.to_csv(os.path.join(OUT_FIG_DIR, f"{promoter_name}_top{TOP_N_HITS}_clusters_summary.csv"), index=False)

    summary = {
        "Promoter": promoter_name,
        "NumHits": len(df),
        "NumFamilies": len(fam_counts),
        "TopFamily": fam_counts.idxmax(),
        "NumClusters": len(cluster_df),
    }
    return summary


# ========= Main: analyse every tfbs_scan file, then aggregate =========

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TFBS_DIR = os.path.join(SCRIPT_DIR, "2.0 data")
    TFBS_PATTERN = "*_tfbs_scan.csv"
    OUT_FIG_DIR = os.path.join(SCRIPT_DIR, "tfbs_analysis_figs")
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    tfbs_files = glob.glob(os.path.join(TFBS_DIR, TFBS_PATTERN))
    if not tfbs_files:
        print(f"ERROR: No files found in {TFBS_DIR}")
        return

    print(f"Found {len(tfbs_files)} TFBS result files.")

    summaries = []

    # Collects the annotated hits of every promoter for the global analysis
    all_hits_list = []

    for csv_path in tfbs_files:
        promoter_name = os.path.basename(csv_path).replace("_tfbs_scan.csv", "")

        # Per-promoter analysis, which also writes the single-promoter figures
        summary = analyze_single_tfbs(csv_path)

        if summary is not None:
            summaries.append(summary)

            # Re-read the annotated table written by analyze_single_tfbs so the
            # global analysis uses exactly the same filtered hits
            detail_csv_path = os.path.join(OUT_FIG_DIR, f"{promoter_name}_top{TOP_N_HITS}_with_family.csv")
            if os.path.exists(detail_csv_path):
                df_detail = pd.read_csv(detail_csv_path)
                df_detail["Promoter"] = promoter_name
                all_hits_list.append(df_detail)

    if not all_hits_list:
        print("No valid data found.")
        return

    big_df = pd.concat(all_hits_list, ignore_index=True)

    # ========= Family distribution across all promoters =========
    print("\nGenerating global family summary plot...")

    global_fam_counts = big_df["Family"].value_counts()

    plt.figure(figsize=(10, 6))
    global_fam_counts.plot(kind="bar", color="darkcyan", edgecolor="black")
    plt.title(f"Global TF Family Distribution (Across all {len(tfbs_files)} promoters)")
    plt.xlabel("TF Family")
    plt.ylabel("Total Number of Hits (Top 100 per promoter)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "GLOBAL_all_promoters_family_counts.png"), dpi=300)
    plt.close()

    # Stacked bar of family composition per promoter, for comparing
    # e.g. hACTB against CMV
    pivot_fam = big_df.groupby(["Promoter", "Family"]).size().unstack(fill_value=0)
    pivot_fam_pct = pivot_fam.div(pivot_fam.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(12, 6))
    pivot_fam_pct.plot(kind="bar", stacked=True, colormap="tab20", figsize=(12, 6))
    plt.title("TF Family Composition by Promoter (%)")
    plt.ylabel("Percentage of Hits")
    plt.xlabel("Promoter")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "GLOBAL_promoter_family_composition.png"), dpi=300)
    plt.close()


    # ========= Detailed table: Promoter | Family | TF | hit count =========
    print("Generating global detailed CSV...")

    global_stats = big_df.groupby(["Promoter", "Family", "TF"]).size().reset_index(name="Hits_Count")

    global_stats = global_stats.sort_values(["Promoter", "Hits_Count"], ascending=[True, False])

    csv_out_path = os.path.join(OUT_FIG_DIR, "GLOBAL_detailed_stats.csv")
    global_stats.to_csv(csv_out_path, index=False)
    print(f"Saved global stats to: {csv_out_path}")

    # ========= Summary metrics across promoters =========
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(os.path.join(OUT_FIG_DIR, "summary_metrics.csv"), index=False)

    plt.figure(figsize=(10, 4))
    plt.bar(sum_df["Promoter"], sum_df["NumClusters"], color="gray")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of clusters")
    plt.title("TFBS clusters per promoter")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "GLOBAL_cluster_counts.png"), dpi=300)
    plt.close()

    print("All analysis done!")

    # ========= Merged statistics, pooling all promoters together =========
    print("\nGenerating GLOBAL MERGED stats (Ignoring Promoter source)...")

    # Total hit count per TF across all promoters
    merged_tf_stats = big_df.groupby(["Family", "TF"]).size().reset_index(name="Total_Hits")
    merged_tf_stats = merged_tf_stats.sort_values("Total_Hits", ascending=False)

    merged_csv_path = os.path.join(OUT_FIG_DIR, "GLOBAL_MERGED_TF_stats.csv")
    merged_tf_stats.to_csv(merged_csv_path, index=False)

    # Most frequent TFs overall
    top_n_plot = 30
    top_tfs = merged_tf_stats.head(top_n_plot)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_tfs, x="TF", y="Total_Hits", hue="Family", dodge=False, palette="viridis")

    plt.title(f"Top {top_n_plot} Most Frequent TFs (Aggregated across all promoters)")
    plt.xlabel("Transcription Factor")
    plt.ylabel("Total Hits Count")
    plt.xticks(rotation=45, ha="right")

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Family")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "GLOBAL_MERGED_top_TFs.png"), dpi=300)
    plt.close()


    # Family distribution overall, one colour per family
    print("Generating colored global family plot...")

    merged_fam_counts = big_df["Family"].value_counts().reset_index()
    merged_fam_counts.columns = ["Family", "Total_Hits"]

    plt.figure(figsize=(14, 7))  # wide enough that the family names do not collide

    # hue="Family" with legend=False colours each bar individually while
    # keeping the names on the x axis only. "turbo" spans a wide colour
    # range; "tab20", "husl" or "Spectral" also work here.
    sns.barplot(
        data=merged_fam_counts,
        x="Family",
        y="Total_Hits",
        hue="Family",
        palette="turbo",
        legend=False,
        edgecolor="black"
    )

    plt.title("Global TF Family Distribution (Aggregated hits)")
    plt.xlabel("TF Family")
    plt.ylabel("Total Hits")

    plt.xticks(rotation=45, ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "GLOBAL_MERGED_family_counts.png"), dpi=300)
    plt.close()

    print("Global merged analysis done!")

if __name__ == "__main__":
    main()