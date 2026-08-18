import pandas as pd
import matplotlib.pyplot as plt
import sys

# ========= 0. Configuration =========
# Path to the TFBS results CSV produced by your scanning script
TFBS_CSV = "Sequence/2.0 data/hACTB promoter_tfbs_scan.csv"
OUT_TOP_CSV = "Sequence/new_hACTB.long_sequences_topTF_hits_Effective.csv"

# Score column used for all filtering and ranking
SCORE_COL = "EffectiveRelScore" 

# Score cutoff for "high-confidence" sites. EffectiveRelScore is distributed
# slightly differently from RelScore, so check the describe() output below
# before tuning this value.
SCORE_THRESHOLD = 0.9 

# Per-TF: keep only top N sites (by Score)
TOP_N_PER_TF = 5

# TSS position on the hACTB promoter sequence (1-based index)
# Example: if your 550 bp sequence is -500..+49 relative to TSS, then TSS_POS = 501
TSS_POS = 501

# Core promoter window relative to TSS (bp)
CORE_UPSTREAM = -500  # e.g. -300 bp upstream
CORE_DOWNSTREAM = 50   # e.g. +50 bp downstream

# Number of top TFs (by high-score / core-window hit count) to show/plot
TOP_N_TFS_PLOT = 30

# Housekeeping-related TF families (simple name-based filter, case-insensitive)
HOUSEKEEPING_FAMILY_KEYWORDS = [
    "SP", "NFY", "TBP", "VEZF", "MAZ", "TFAP"
]

# ========= 1. Load results and basic overview =========
try:
    df = pd.read_csv(TFBS_CSV)
except FileNotFoundError:
    print(f"Error: File {TFBS_CSV} not found.")
    sys.exit(1)

print("=== All TFBS hits: basic info ===")
print(df.head())
print("Total hits:", len(df))

# The score column must be present
if SCORE_COL not in df.columns:
    print(f"Error: Column '{SCORE_COL}' not found in input CSV. Available columns: {list(df.columns)}")
    # Optional fallback: SCORE_COL = "RelScore"
    sys.exit(1)

print(f"\n{SCORE_COL} distribution (all hits):")
print(df[SCORE_COL].describe())

# ========= 2. Global high-score filter =========
df_high = df[df[SCORE_COL] >= SCORE_THRESHOLD].copy()

print(f"\n=== High-score sites ({SCORE_COL} >= {SCORE_THRESHOLD}) ===")
print("High-score hits:", len(df_high))

if len(df_high) == 0:
    print(f"WARNING: No hits found with {SCORE_COL} >= {SCORE_THRESHOLD}.")
    print("Consider lowering the SCORE_THRESHOLD.")
    sys.exit(0)

# ========= 3. Per-TF top N filtering =========
df_high = (
    df_high.sort_values(["TF", SCORE_COL], ascending=[True, False])
    .groupby("TF")
    .head(TOP_N_PER_TF)
    .reset_index(drop=True)
)

print(f"\nAfter per-TF top{TOP_N_PER_TF} filter:")
print("Hits remaining:", len(df_high))
print("TF counts (top 20):")
print(df_high["TF"].value_counts().head(20))

# ========= 4. Distance to TSS and core-promoter window =========
df_high["distance_to_TSS"] = df_high["Start"] - TSS_POS

# Core window: e.g. [-300, +50] relative to TSS
core_mask = (df_high["distance_to_TSS"] >= CORE_UPSTREAM) & (
    df_high["distance_to_TSS"] <= CORE_DOWNSTREAM
)
df_core = df_high[core_mask].copy()

print(
    f"\nCore promoter window: [{CORE_UPSTREAM}, {CORE_DOWNSTREAM}] bp "
    f"around TSS (position {TSS_POS})"
)
print("Hits in core window:", len(df_core))

if len(df_core) == 0:
    print("WARNING: No hits in the specified core promoter window.")
else:
    print(f"\nExample core-window high-score sites (top 20 by {SCORE_COL}):")
    print(
        df_core[
            ["TF", "Start", "End", "Strand", SCORE_COL, "distance_to_TSS"]
        ].sort_values(SCORE_COL, ascending=False).head(20)
    )

# ========= 5. Rank TFs (Modified Logic: Elite Score Priority) =========
if len(df_core) > 0:
    # Per TF: best score and number of hits inside the core window
    tf_stats = df_core.groupby("TF")[SCORE_COL].agg(['max', 'count']).reset_index()
    tf_stats.columns = ['TF', 'max_score', 'core_hits']

    # Flag housekeeping-like TF families
    def is_housekeeping_like(tf_name: str) -> bool:
        tf_name = str(tf_name).upper()
        return any(k in tf_name for k in HOUSEKEEPING_FAMILY_KEYWORDS)
    
    tf_stats['is_housekeeping'] = tf_stats['TF'].apply(is_housekeeping_like)

    # Flag "elite" TFs, i.e. those with an exceptionally high best score
    ELITE_SCORE_THRESHOLD = 1.185
    tf_stats['is_elite'] = tf_stats['max_score'] >= ELITE_SCORE_THRESHOLD

    # Ranking priority:
    #   1. elite TFs first
    #   2. then housekeeping-like families
    #   3. then higher max_score
    #   4. then more hits in the core window
    tf_rank = tf_stats.sort_values(
        by=['is_elite', 'is_housekeeping', 'max_score', 'core_hits'],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    print("\n=== Ranked TFs in core window (Elite Score > 1.1 Prioritized) ===")
    print(tf_rank.head(30))
    
    # Report which TFs were promoted by the elite rule
    elites = tf_rank[tf_rank['is_elite'] == True]
    if not elites.empty:
        print(f"\n[INFO] {len(elites)} Elite TFs found (Score >= {ELITE_SCORE_THRESHOLD}):")
        print(elites[['TF', 'max_score', 'core_hits', 'is_housekeeping']])
else:
    tf_rank = None

# ========= 6. Plot: top TFs in core window =========
if len(df_core) > 0:
    # Decide which TFs to plot
    if tf_rank is not None and not tf_rank.empty:
        top_tfs_for_plot = tf_rank["TF"].head(TOP_N_TFS_PLOT).tolist()
    else:
        top_tfs_for_plot = (
            df_core["TF"].value_counts().head(TOP_N_TFS_PLOT).index.tolist()
        )

    sub = df_core[df_core["TF"].isin(top_tfs_for_plot)]

    print(f"\nTop {TOP_N_TFS_PLOT} TFs used for plotting (core window):")
    print(sub["TF"].value_counts())

    # 40 hand-picked high-contrast colours for categorical data, chosen so
    # that any two TF series stay visually distinguishable
    DISTINCT_COLORS = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
        '#ffffff', '#000000', '#1f77b4', '#ff7f0e', '#2ca02c',
        '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a',
        '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d'
    ]
    
    num_needed = len(top_tfs_for_plot)
    
    # Fall back to a continuous colormap if more than 40 colours are needed
    if num_needed > len(DISTINCT_COLORS):
        import numpy as np
        colors = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, num_needed))
    else:
        colors = DISTINCT_COLORS[:num_needed]
    
    tf_color_map = dict(zip(top_tfs_for_plot, colors))

    plt.figure(figsize=(15, 8))  # extra height for the multi-column legend
    
    for tf in top_tfs_for_plot:
        sub_tf = sub[sub["TF"] == tf]
        plt.scatter(
            sub_tf["Start"],
            sub_tf[SCORE_COL], 
            label=tf, 
            s=70,
            alpha=0.9,
            color=tf_color_map[tf], 
            edgecolors='black',  # outlines keep similar colours apart
            linewidth=0.6
        )

    # Mark TSS
    plt.axvline(TSS_POS, color="k", linestyle="--", linewidth=1.5, label="TSS")
    
    plt.xlabel("Position on hACTB promoter (bp)", fontsize=12)
    plt.ylabel(f"Score ({SCORE_COL})", fontsize=12)
    plt.title(
        f"hACTB promoter TFBS "
        f"({SCORE_COL} >= {SCORE_THRESHOLD}, per-TF top{TOP_N_PER_TF}, "
        f"core [{CORE_UPSTREAM},{CORE_DOWNSTREAM}] bp)", fontsize=14
    )
    plt.grid(True, linestyle=':', alpha=0.5)

    # Legend order: TSS first, then TF names alphabetically
    handles, labels = plt.gca().get_legend_handles_labels()
    
    tss_item = [(h, l) for h, l in zip(handles, labels) if l == "TSS"]
    tf_items = [(h, l) for h, l in zip(handles, labels) if l != "TSS"]
    
    tf_items.sort(key=lambda x: x[1])
    
    final_items = tss_item + tf_items
    
    if final_items:
        final_handles, final_labels = zip(*final_items)
        plt.legend(
            final_handles, final_labels, 
            bbox_to_anchor=(1.01, 1), 
            loc="upper left", 
            fontsize=9, 
            title="TF Name",
            ncol=2,
            frameon=True,
            edgecolor='black'
        )

    plt.tight_layout()
    plt.show()

    print("\nExample rows from the plotted subset:")
    print(
        sub[
            ["TF", "Start", "End", "Strand", SCORE_COL, "distance_to_TSS"]
        ].sort_values(["TF", SCORE_COL], ascending=[True, False]).head(20)
    )

    # ========= 7. Export to CSV =========
    export_cols = [
        "TF", "Start", "End", "Strand",
        "EffectiveRelScore", "RelScore", "RawScore",
        "SiteSeq", "distance_to_TSS", "Motif_ID"
    ]
    export_cols = [c for c in export_cols if c in sub.columns]

    sub_export = sub.sort_values(
        by=["Start", "TF", SCORE_COL],
        ascending=[True, True, False],
    )[export_cols]

    sub_export.to_csv(OUT_TOP_CSV, index=False)
    print(f"\nTop TF core-window hits written to:")
    print(OUT_TOP_CSV)

else:
    print("\nNo core-window hits to plot.")