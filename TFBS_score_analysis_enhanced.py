import pandas as pd
import matplotlib.pyplot as plt
import sys

# ========= 0. Configuration =========
# Path to the TFBS results CSV produced by your scanning script
TFBS_CSV = "Sequence/2.0 data/hACTB promoter_tfbs_scan.csv"
OUT_TOP_CSV = "Sequence/new_hACTB.long_sequences_topTF_hits_Effective.csv"

# [修改] 设置主要筛选分数列
SCORE_COL = "EffectiveRelScore" 

# [修改] Score cutoff for "high-confidence" sites based on EffectiveRelScore
# 注意：EffectiveRelScore 的分布可能与 RelScore 略有不同，建议检查 describe() 输出后微调此值
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

# 检查是否存在 EffectiveRelScore 列
if SCORE_COL not in df.columns:
    print(f"Error: Column '{SCORE_COL}' not found in input CSV. Available columns: {list(df.columns)}")
    # 如果没有 EffectiveRelScore，回退到 RelScore (可选)
    # SCORE_COL = "RelScore"
    sys.exit(1)

print(f"\n{SCORE_COL} distribution (all hits):")
print(df[SCORE_COL].describe())

# ========= 2. Global high-score filter =========
# [修改] 使用 SCORE_COL 进行筛选
df_high = df[df[SCORE_COL] >= SCORE_THRESHOLD].copy()

print(f"\n=== High-score sites ({SCORE_COL} >= {SCORE_THRESHOLD}) ===")
print("High-score hits:", len(df_high))

if len(df_high) == 0:
    print(f"WARNING: No hits found with {SCORE_COL} >= {SCORE_THRESHOLD}.")
    print("Consider lowering the SCORE_THRESHOLD.")
    sys.exit(0)

# ========= 3. Per-TF top N filtering =========
# [修改] 根据 SCORE_COL 进行排序
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
    # 1. 统计每个 TF 的核心数据：最高分 (Max Score) 和 总次数 (Hits)
    # 使用 agg 函数同时计算 max 和 count
    tf_stats = df_core.groupby("TF")[SCORE_COL].agg(['max', 'count']).reset_index()
    tf_stats.columns = ['TF', 'max_score', 'core_hits']

    # 2. 标记是否为 Housekeeping 基因
    def is_housekeeping_like(tf_name: str) -> bool:
        tf_name = str(tf_name).upper()
        return any(k in tf_name for k in HOUSEKEEPING_FAMILY_KEYWORDS)
    
    tf_stats['is_housekeeping'] = tf_stats['TF'].apply(is_housekeeping_like)

    # 3. [关键修改] 标记是否为 "Elite High Score" (特别高分)
    # 设定阈值为 1.15 (根据你的 ZNF70 分数设定)
    ELITE_SCORE_THRESHOLD = 1.185
    tf_stats['is_elite'] = tf_stats['max_score'] >= ELITE_SCORE_THRESHOLD

    # 4. [关键修改] 排序逻辑
    # 优先级顺序：
    #   1. is_elite (True): 最高分 > 1.18 的，强制排在最前面！
    #   2. is_housekeeping (True): 其次是管家基因
    #   3. max_score (High->Low): 同梯队里，分数高的排前面
    #   4. core_hits (High->Low): 分数也一样，数量多的排前面
    
    tf_rank = tf_stats.sort_values(
        by=['is_elite', 'is_housekeeping', 'max_score', 'core_hits'],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    print("\n=== Ranked TFs in core window (Elite Score > 1.1 Prioritized) ===")
    print(tf_rank.head(30))
    
    # 打印出是谁因为高分被“提拔”了
    elites = tf_rank[tf_rank['is_elite'] == True]
    if not elites.empty:
        print(f"\n[INFO] {len(elites)} Elite TFs found (Score >= {ELITE_SCORE_THRESHOLD}):")
        print(elites[['TF', 'max_score', 'core_hits', 'is_housekeeping']])
else:
    tf_rank = None

# ========= 6. Plot: top TFs in core window =========
if len(df_core) > 0:
    # 确定要画哪些 TF
    if tf_rank is not None and not tf_rank.empty:
        top_tfs_for_plot = tf_rank["TF"].head(TOP_N_TFS_PLOT).tolist()
    else:
        top_tfs_for_plot = (
            df_core["TF"].value_counts().head(TOP_N_TFS_PLOT).index.tolist()
        )

    # 筛选数据
    sub = df_core[df_core["TF"].isin(top_tfs_for_plot)]

    print(f"\nTop {TOP_N_TFS_PLOT} TFs used for plotting (core window):")
    print(sub["TF"].value_counts())

    # --- [关键修改] 使用40种人工挑选的高对比度颜色 ---
    # 这组颜色是专门为分类数据设计的，确保任意两种颜色都有显著差异
    # 包含：红, 绿, 黄, 蓝, 橙, 紫, 青, 品红, 莱姆, 粉, 蓝绿, 薰衣草, 棕, 米, 栗, 薄荷, 橄榄, 杏, 海军蓝, 灰...
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
    
    # 如果需要的颜色超过了列表长度，循环使用（但通常前40个已经足够覆盖绝大多数情况）
    if num_needed > len(DISTINCT_COLORS):
        import numpy as np
        # 极端情况：回到 nipy_spectral
        colors = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, num_needed))
    else:
        colors = DISTINCT_COLORS[:num_needed]
    
    # 分配颜色
    tf_color_map = dict(zip(top_tfs_for_plot, colors))
    # ----------------------------------------

    plt.figure(figsize=(15, 8)) # 画布加高，容纳多行图例
    
    # 循环画图
    for tf in top_tfs_for_plot:
        sub_tf = sub[sub["TF"] == tf]
        plt.scatter(
            sub_tf["Start"],
            sub_tf[SCORE_COL], 
            label=tf, 
            s=70,  # 点大小
            alpha=0.9, # 高不透明度，让颜色更实
            color=tf_color_map[tf], 
            edgecolors='black', # 黑色边框是区分相似颜色的关键！
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

    # --- 图例排序：TSS置顶 + 其余按名称排序 ---
    handles, labels = plt.gca().get_legend_handles_labels()
    
    tss_item = [(h, l) for h, l in zip(handles, labels) if l == "TSS"]
    tf_items = [(h, l) for h, l in zip(handles, labels) if l != "TSS"]
    
    tf_items.sort(key=lambda x: x[1])
    
    final_items = tss_item + tf_items
    
    if final_items:
        final_handles, final_labels = zip(*final_items)
        # ncol=3: 30个颜色分3列排，这样图例比较方正，容易看
        plt.legend(
            final_handles, final_labels, 
            bbox_to_anchor=(1.01, 1), 
            loc="upper left", 
            fontsize=9, 
            title="TF Name",
            ncol=2,    # 分2列显示
            frameon=True,
            edgecolor='black'
        )
    # --------------------------------------------------------

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