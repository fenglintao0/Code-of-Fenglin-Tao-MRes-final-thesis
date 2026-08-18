import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========= 配置 =========

TFBS_DIR = "Sequence/2.0 data"
TFBS_PATTERN = "*_tfbs_scan.csv"

WINDOW_SIZE = 100
WINDOW_STEP = 20

TOP_N_HITS = 100  # 你现在用的是 top100

OUT_FIG_DIR = "Sequence/tfbs_analysis_figs"
os.makedirs(OUT_FIG_DIR, exist_ok=True)


# ========= TF family 映射 =========
def assign_family(tf: str) -> str:
    t = tf.upper()

    # ==========================================
    # 1. 您特别关注 / 之前漏掉的特定 TF (Priority)
    # ==========================================
    if "YY" in t: return "YY1 Family"             # YY1, YY2 (Zinc Finger)
    if "NFI" in t: return "NFI"                   # NFIX, NFIA, NFIB, NFIC (CCAAT-binding)
    if "NFAT" in t: return "NFAT"                 # NFATC1-4 (Immune)
    if "GMEB" in t: return "GMEB"                 # GMEB1/2 (Zinc Finger)
    if "ZGLP" in t: return "GATA-like (ZGLP)"     # GATA-like Zinc Finger
    
    # ==========================================
    # 2. 核心关注：GC-rich / Zinc Fingers (细分)
    # ==========================================
    if "KLF" in t: return "KLF"
    if t.startswith("SP") and len(t) <= 4: return "SP"
    if "MAZ" in t or "VEZF" in t: return "SP-like"
    if "HIC" in t: return "HIC (ZF)"
    if "PLAG" in t: return "PLAG (ZF)"
    if "CTCF" in t: return "CTCF"                 # 绝缘子
    if "ZEB" in t or "SNAI" in t or "SLUG" in t: return "EMT-TF (ZF)"
    if "EGR" in t or "WT1" in t: return "EGR/WT1"
    
    # 广义 Zinc Finger 捕获 (ZNF系列，数量巨大)
    # 放在这里是为了把有具体名字的 ZF (如 GATA) 先分出去
    zf_keywords = ["ZNF", "ZBTB", "ZSCAN", "ZFX", "ZIC", "ZFP", "ZKSCAN", "ZIM", "ZBED"]
    if any(k in t for k in zf_keywords): return "Zinc Finger (General)"

    # ==========================================
    # 3. 大型结构家族 (覆盖面广)
    # ==========================================
    
    # ETS 家族 (非常大)
    # ELF, ELK, ETS, ERG, FLI, GABPA, SPI, PU.1, ETV
    ets_keywords = ["ETS", "ELK", "ELF", "ERG", "FLI", "GABPA", "SPI", "ETV", "FEV"]
    if any(k in t for k in ets_keywords): return "ETS"

    # Homeobox (同源异形盒)
    # HOX, PAX, SOX, POU, DLX, LHX, NKX, SIX, MEIS, PBX, OTX
    homeo_keywords = ["HOX", "PAX", "SOX", "POU", "DLX", "LHX", "NKX", "SIX", 
                      "MEIS", "PBX", "OTX", "NANOG", "OCT4", "CDX", "MIX", "ISL", "GBX"]
    if any(k in t for k in homeo_keywords): return "Homeobox"

    # bHLH (碱性螺旋-环-螺旋)
    # MYC, MAX, USF, TCF, TWIST, NEURO, ASCL, HIF, HAND, HEY, ID
    bhlh_keywords = ["MYC", "MAX", "USF", "TCF", "TWIST", "NEURO", "ASCL", 
                     "HIF", "HAND", "HEY", "BHLH", "NHLH", "OLIG"]
    if any(k in t for k in bhlh_keywords): return "bHLH"

    # bZIP (碱性亮氨酸拉链)
    # CREB, ATF, AP-1 (FOS/JUN), CEBP, MAF, NFE2, BACH
    if any(k in t for k in ["CREB", "ATF"]): return "CREB/ATF"
    if any(k in t for k in ["FOS", "JUN", "AP1"]): return "AP-1"
    if "CEBP" in t: return "C/EBP"
    if any(k in t for k in ["MAF", "NFE2", "BACH", "DBP", "HLF", "TEF"]): return "bZIP (Other)"

    # Forkhead (FOX 家族)
    if "FOX" in t: return "Forkhead (FOX)"

    # GATA 家族
    if "GATA" in t: return "GATA"

    # Nuclear Receptors (核受体)
    # NR, ESR, AR, RAR, RXR, PPAR, VDR, GR, PR, ROR, EAR
    nr_keywords = ["NR", "ESR", "AR", "RAR", "RXR", "PPAR", "VDR", "GR", "PR", "ROR", "HNF4", "COUP"]
    if any(k in t for k in nr_keywords) or t.startswith("NR"): return "Nuclear Receptor"

    # ==========================================
    # 4. 其他重要功能家族
    # ==========================================
    
    # E2F
    if t.startswith("E2F") or "TFDP" in t: return "E2F"
    
    # AP-2
    if "TFAP2" in t or "AP-2" in t: return "AP-2"
    
    # STAT / IRF (免疫)
    if "STAT" in t: return "STAT"
    if "IRF" in t: return "IRF"
    if "REL" in t or "NFKB" in t: return "NF-kB"

    # Runt (RUNX)
    if "RUNX" in t: return "Runt (RUNX)"

    # T-box (TBX)
    if "TBX" in t or "TBR" in t or "EOMES" in t: return "T-box"

    # High Mobility Group (HMG) - 除了 SOX 以外的
    if "LEF" in t or "TCF" in t or "HMG" in t: return "HMG"

    # p53
    if "TP53" in t or "P53" in t or "P63" in t or "P73" in t: return "p53 Family"
    
    # TBP / GTF
    if "TBP" in t or "GTF" in t: return "Basal Machinery"
    
    # MADS box
    if "SRF" in t or "MEF2" in t: return "MADS box"

    # TEAD (Hippo pathway)
    if "TEAD" in t: return "TEAD"

    # ==========================================
    # 5. 最后兜底
    # ==========================================
    return "Other"


# ========= 对单个 promoter 分析 =========

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

    # 一维位置聚类
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

    # ===== 可视化（单 promoter）=====

    # family counts
    plt.figure(figsize=(8, 4))
    fam_counts.sort_values(ascending=False).plot(kind="bar", color="steelblue")
    plt.ylabel("Number of TFBS")
    plt.title(f"{promoter_name}: TF family counts (top {TOP_N_HITS} hits)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, f"{promoter_name}_family_counts.png"), dpi=300)
    plt.close()

    # density
    plt.figure(figsize=(10, 4))
    centers = (dens_df["WindowStart"] + dens_df["WindowEnd"]) / 2.0
    plt.plot(centers, dens_df["Count"], marker="o", linestyle="-")
    plt.xlabel("Position on promoter (bp)")
    plt.ylabel(f"TFBS count in {WINDOW_SIZE} bp window")
    plt.title(f"{promoter_name}: TFBS density (top {TOP_N_HITS} hits)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, f"{promoter_name}_density.png"), dpi=300)
    plt.close()

    # density by family heatmap
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

    # cluster schematic
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

    # 保存带 family 的 TFBS 表以及 cluster summary
    df.to_csv(os.path.join(OUT_FIG_DIR, f"{promoter_name}_top{TOP_N_HITS}_with_family.csv"), index=False)
    cluster_df.to_csv(os.path.join(OUT_FIG_DIR, f"{promoter_name}_top{TOP_N_HITS}_clusters_summary.csv"), index=False)

    # 返回用于全局汇总的信息
    summary = {
        "Promoter": promoter_name,
        "NumHits": len(df),
        "NumFamilies": len(fam_counts),
        "TopFamily": fam_counts.idxmax(),
    "NumClusters": len(cluster_df),
    }
    return summary


# ========= 主程序：遍历所有 tfbs_scan，并做全局汇总 =========

# ========= 主程序：遍历所有 tfbs_scan，并做全局汇总 =========

def main():
    # 1. 路径检查（保留之前的绝对路径逻辑）
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
    
    # === 新增：用于存储所有启动子的所有 TFBS 数据 ===
    all_hits_list = [] 

    for csv_path in tfbs_files:
        # 获取 promoter 名字
        promoter_name = os.path.basename(csv_path).replace("_tfbs_scan.csv", "")
        
        # 1. 调用单文件分析函数（这一步生成单张图）
        summary = analyze_single_tfbs(csv_path)
        
        if summary is not None:
            summaries.append(summary)
            
            # 2. 读取刚刚生成的带 Family 的中间文件，或者重新读原始数据
            # 为了简单，我们直接利用 analyze_single_tfbs 里保存的 "top100_with_family.csv"
            # 这样保证逻辑一致
            detail_csv_path = os.path.join(OUT_FIG_DIR, f"{promoter_name}_top{TOP_N_HITS}_with_family.csv")
            if os.path.exists(detail_csv_path):
                df_detail = pd.read_csv(detail_csv_path)
                df_detail["Promoter"] = promoter_name # 给数据打上标签
                all_hits_list.append(df_detail)

    # 如果没有任何数据，直接退出
    if not all_hits_list:
        print("No valid data found.")
        return

    # === 合并所有数据 ===
    big_df = pd.concat(all_hits_list, ignore_index=True)

    # ==========================================
    # 需求 1: 全部 Promoter 的家族分析条形图
    # ==========================================
    print("\nGenerating global family summary plot...")
    
    # 统计所有启动子加在一起，各个家族出现了多少次
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

    # (可选) 如果你想看每个启动子的家族比例堆叠图 (Stacked Bar)
    # 这张图能直观对比 hACTB 和 CMV 在家族构成上的区别
    pivot_fam = big_df.groupby(["Promoter", "Family"]).size().unstack(fill_value=0)
    # 归一化为百分比
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


    # ==========================================
    # 需求 2: 完整 CSV (家族 + 名字 + hits数量)
    # ==========================================
    print("Generating global detailed CSV...")

    # 我们做一个统计表：Promoter | Family | TF | Count
    # 例如: hACTB | Sp | Sp1 | 12
    global_stats = big_df.groupby(["Promoter", "Family", "TF"]).size().reset_index(name="Hits_Count")
    
    # 排序：先按 Promoter 排，再按 Hits 数量降序排
    global_stats = global_stats.sort_values(["Promoter", "Hits_Count"], ascending=[True, False])

    # 保存
    csv_out_path = os.path.join(OUT_FIG_DIR, "GLOBAL_detailed_stats.csv")
    global_stats.to_csv(csv_out_path, index=False)
    print(f"Saved global stats to: {csv_out_path}")

    # ==========================================
    # (原有功能) 简单的汇总图
    # ==========================================
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(os.path.join(OUT_FIG_DIR, "summary_metrics.csv"), index=False)
    
    # 简单的 Cluster 数量对比
    plt.figure(figsize=(10, 4))
    plt.bar(sum_df["Promoter"], sum_df["NumClusters"], color="gray")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of clusters")
    plt.title("TFBS clusters per promoter")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "GLOBAL_cluster_counts.png"), dpi=300)
    plt.close()

    print("All analysis done!")

    # big_df = pd.concat(all_hits_list, ignore_index=True)

    # ==========================================
    # 需求: 全局合并统计 (不区分 Promoter)
    # ==========================================
    print("\nGenerating GLOBAL MERGED stats (Ignoring Promoter source)...")

    # 1. 统计每个 TF 在所有启动子中总共出现了多少次
    #    Grouping by Family and TF name
    merged_tf_stats = big_df.groupby(["Family", "TF"]).size().reset_index(name="Total_Hits")
    
    #    Sort by Total_Hits descen(降序排列)
    merged_tf_stats = merged_tf_stats.sort_values("Total_Hits", ascending=False)

    # 2. 保存这个“英雄榜”表格
    merged_csv_path = os.path.join(OUT_FIG_DIR, "GLOBAL_MERGED_TF_stats.csv")
    merged_tf_stats.to_csv(merged_csv_path, index=
    
    # ==========================================
    # 绘图 1: 全局最常见的 Top 30 TF (Bar Plot)
    # ==========================================
    top_n_plot = 30
    top_tfs = merged_tf_stats.head(top_n_plot)

    plt.figure(figsize=(12, 6))
    # 使用 seaborn 画图
    sns.barplot(data=top_tfs, x="TF", y="Total_Hits", hue="Family", dodge=False, palette="viridis")
    
    plt.title(f"Top {top_n_plot} Most Frequent TFs (Aggregated across all promoters)")
    plt.xlabel("Transcription Factor")
    plt.ylabel("Total Hits Count")
    plt.xticks(rotation=45, ha="right")
    
    # 调整图例位置
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Family")
    
    # === 这里是刚才报错的地方，修正为 tight_layout() ===
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "GLOBAL_MERGED_top_TFs.png"), dpi=300)
    plt.close()   
    
    
    # ==========================================
    # 绘图 2: 全局家族分布 (彩色版)
    # ==========================================
    print("Generating colored global family plot...")

    # 1. 准备数据：把 Series 转成 DataFrame，方便 Seaborn 画图
    merged_fam_counts = big_df["Family"].value_counts().reset_index()
    merged_fam_counts.columns = ["Family", "Total_Hits"]

    plt.figure(figsize=(14, 7)) # 把画布稍微调宽一点，防止家族名字挤在一起

    # 2. 使用 Seaborn 画图
    # palette="turbo": 一种色彩跨度很大的调色板，适合区分很多个柱子
    # hue="Family": 告诉 Seaborn 根据 Family 的名字来分配颜色
    # legend=False: 不需要图例，因为 X 轴标签已经写了名字
    sns.barplot(
        data=merged_fam_counts, 
        x="Family", 
        y="Total_Hits", 
        hue="Family", 
        palette="turbo",  # 您也可以换成 "tab20", "husl", "Spectral"
        legend=False,
        edgecolor="black" # 给柱子加个黑边，更清晰
    )

    plt.title("Global TF Family Distribution (Aggregated hits)")
    plt.xlabel("TF Family")
    plt.ylabel("Total Hits")
    
    # 3. 调整 X 轴标签，防止重叠
    plt.xticks(rotation=45, ha="right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "GLOBAL_MERGED_family_counts.png"), dpi=300)
    plt.close()

    print("Global merged analysis done!")

if __name__ == "__main__":
    main()