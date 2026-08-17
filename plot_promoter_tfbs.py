import os
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

# ========= 配置区（请按需要修改） =========

# Promoter FASTA 文件
FASTA_PATH = "Sequence/promoter/hACTB promoter.fasta"

# TFBS 结果文件（Excel 或 CSV）
TFBS_PATH = "Sequence/5.0 data/TFAP2.xlsx"  # 支持.xlsx 或.csv

# TSS 位置（1-based，与 TFBS Start/End 同一坐标系统）
TSS_POS = 501  # 根据你的分析脚本设置

# 如果只想画某些 TF，可以填入列表；None 表示全部
TF_FILTER = None
# 例如：TF_FILTER = ["TFAP2A", "TFAP2B", "TFAP2C", "KLF5", "SP3", "E2F1"]

# 是否区分正负链，True 时 +、- 绘制在同一个 TF 行内上下错开
SEPARATE_STRANDS = True

# 图片输出文件名
OUTPUT_FIG = "Sequence/hACTB_promoter_TFBS.png"

# 特定 motif/区域标注（1-based 坐标）
Synthetic_Part = (143, 276)
CCAAT_REGION = (410, 414)
TATA_REGION  = (472, 477)
GC_RICH_REGION = (452, 471)
PolyAT_tracts= (327, 337)

# ========= 1. 读入 promoter 序列 =========

def load_promoter_length(fasta_path):
    record = next(SeqIO.parse(fasta_path, "fasta"))
    seq_len = len(record.seq)
    print(f"Loaded promoter: {record.id}, length = {seq_len} bp")
    return seq_len

promoter_len = load_promoter_length(FASTA_PATH)

# ========= 2. 读入 TFBS 表（Excel/CSV） =========

def load_tfbs_table(tfbs_path):
    ext = os.path.splitext(tfbs_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(tfbs_path)
    else:
        df = pd.read_csv(tfbs_path)

    # 统一列名（按需要扩展）
    col_map = {
        "tf": "TF",
        "motif": "TF"
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    required_cols = ["TF", "Start", "End"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"TFBS table must contain column '{c}'")

    # 没有 Strand 列就默认全部 "+"
    if "Strand" not in df.columns:
        df["Strand"] = "+"

    return df

df = load_tfbs_table(TFBS_PATH)
print(f"Loaded TFBS hits: {len(df)}")

# 如有 TF_FILTER，只保留指定 TF
if TF_FILTER is not None:
    df = df[df["TF"].isin(TF_FILTER)].copy()
    print(f"After TF filter: {len(df)} hits, TFs = {sorted(df['TF'].unique())}")

if df.empty:
    raise SystemExit("No TFBS hits to plot.")

# ========= 3. 准备绘图数据 =========

unique_tfs = sorted(df["TF"].unique())
tf_to_y = {tf: i for i, tf in enumerate(unique_tfs)}

def get_y(tf, strand):
    base_y = tf_to_y[tf]
    if not SEPARATE_STRANDS:
        return base_y
    return base_y + (0.25 if strand == "+" else -0.25)

# ========= 2. 定制颜色：Tab20 优化版 (保留18色 + 补充中间色) =========

TAB20_OPTIMIZED = [
    # --- 1. Tab20 深色组 (最清晰，优先分配) ---
    "#1f77b4", # 深蓝
    "#ff7f0e", # 深橙
    "#2ca02c", # 深绿
    "#d62728", # 深红
    "#9467bd", # 深紫
    "#8c564b", # 深棕
    "#e377c2", # 深粉
    "#aec7e8", # 浅蓝
    "#ff9896", # 浅红
    "#ffbb78", # 浅橙
    "#7f7f7f", # 深灰
    "#bcbd22", # 橄榄
    "#17becf", # 深青


    # --- A组: 完美复刻你提供的参考图 (从深红到深蓝) ---

    "#AE2012", # (Ref 9)  砖红
    "#CA6702", # (Ref 7)  南瓜橙
    "#EE9B00", # (Ref 6)  姜黄/芥末 (不刺眼)
    "#005F73", # (Ref 2)  深孔雀蓝
    "#0A9396", # (Ref 3)  深青色
    "#001219", # (Ref 1)  午夜蓝黑
    "#7570B3", # 蓝紫 (偏莫兰迪)
    "#E7298A", # 洋红
    "#1B9E77", # 深青绿
    
    # --- B组: 按照同等风格补充的深色 (补全紫色/绿色/粉色系) ---
    "#6A4C93", # 皇家紫 (Deep Purple) - 参考图缺少的
    "#386641", # 猎人绿 (Hunter Green) - 参考图缺少的
    "#BC6C25", # 皮革棕 (Leather)
    "#9D0208", # 血红 (Blood Red)
    "#457B9D", # 钢蓝 (Steel Blue)
    "#1D3557", # 海军蓝 (Navy)
    "#7209B7", # 蓝紫 (Violet)
    "#F72585", # 深洋红 (Deep Magenta)
    "#2A9D8F", # 丛林绿 (Jungle Green
]

unique_tfs = sorted(df["TF"].unique())
tf_to_y = {tf: i for i, tf in enumerate(unique_tfs)}

# 分配颜色
tf_to_color = {}
for i, tf in enumerate(unique_tfs):
    tf_to_color[tf] = TAB20_OPTIMIZED[i % len(TAB20_OPTIMIZED)]

unique_tfs = sorted(df["TF"].unique())
tf_to_y = {tf: i for i, tf in enumerate(unique_tfs)}

# 分配颜色
tf_to_color = {}
for i, tf in enumerate(unique_tfs):
    tf_to_color[tf] = TAB20_OPTIMIZED[i % len(TAB20_OPTIMIZED)]


# ========= 4. 绘制图像 =========

fig_height = max(3, 0.43 * len(unique_tfs))  # 每个 TF 行多一点高度
fig, ax = plt.subplots(figsize=(20, fig_height))

for _, row in df.iterrows():
    tf = row["TF"]
    start = int(row["Start"])
    end = int(row["End"])
    strand = str(row.get("Strand", "+"))
    y = get_y(tf, strand)
    color = tf_to_color[tf]

    width = end - start + 1

    # 矩形表示 TFBS
    rect = mpatches.Rectangle(
        (start, y - 0.18),   # 左下角 (x, y)
        width,
        0.36,                # 高度
        facecolor=color,
        edgecolor="black",
        alpha=0.8
    )
    ax.add_patch(rect)

    # 在矩形上方标注 TF 名（颜色与矩形一致）
    ax.text(
        start + width / 2,
        y + 0.18,
        tf,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="bottom",
        rotation=0,
        color=color
    )

# ========= 标注 CCAAT box / TATA box / GC-rich 区 =========

def draw_vertical_region(start, end, label, color, y_pos=None):
    """
    画一个从底到顶的半透明竖条，并在指定 y 位置标注文本。
    y_pos: 文本的 y 坐标（在数据坐标系中），默认取所有 TF 行的中间。
    """
    # 竖条：贯穿整个 y 轴范围
    ax.axvspan(
        start,
        end,
        ymin=0.0,
        ymax=1.0,
        facecolor=color,
        alpha=0.1,
        edgecolor="none",
        zorder=0
    )

    # 文本 y 位置：如果没指定，就取中间一行附近
    if y_pos is None:
        y_pos = (len(unique_tfs) - 1) / 1.5  

    ax.text(
        (start + end) / 2,
        y_pos,
        label,
        fontsize=12,
        ha="center",
        va="center",
        color="black",
        rotation=90   # 垂直文字，如果你想水平就改成 0
    )

# 调用：这里不传 y_pos，默认放在中间；也可以自己给一个数，例如 y_pos=0 或 y_pos=2
draw_vertical_region(CCAAT_REGION[0], CCAAT_REGION[1], "CCAAT BOX", "blue")
draw_vertical_region(TATA_REGION[0],  TATA_REGION[1],  "TATA BOX",  "darkorange")
draw_vertical_region(GC_RICH_REGION[0], GC_RICH_REGION[1], "GC BOX", "green")
draw_vertical_region(PolyAT_tracts[0], PolyAT_tracts[1], "PolyAT_tracts", "purple")
draw_vertical_region(Synthetic_Part[0], Synthetic_Part[1], "Synthetic_Part", "yellow")

# 画 TSS 垂直线
if TSS_POS is not None:
    ax.axvline(TSS_POS, color="red", linestyle="--", linewidth=1.5, label="TSS")

# 坐标轴范围（这里画全长，也可以改成只画 TSS 附近）
x_min, x_max = 1, promoter_len
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1, len(unique_tfs))
ax.set_xlabel("Position on promoter (bp)", fontsize=12)


# 彩色 y 轴标签
ax.set_yticks([tf_to_y[tf] for tf in unique_tfs])
ax.set_yticklabels([""] * len(unique_tfs))  # 先清空

for tf in unique_tfs:
    y = tf_to_y[tf]
    color = tf_to_color[tf]
    ax.text(
        -0.003,                  # 在 y 轴附近
        y,
        tf,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="center",
        color=color,
        transform=ax.get_yaxis_transform()  # y 按轴坐标，x 按轴坐标
    )


# 标题
title = "TFBS distribution on hACTB-TFAP2 family promoter"
if TSS_POS is not None:
    title += f" (TSS at {TSS_POS})"
ax.set_title(title, fontsize=18)

# 图例
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.show()

print(f"Figure saved to: {OUTPUT_FIG}")