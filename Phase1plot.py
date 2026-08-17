import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib import rcParams
from matplotlib.lines import Line2D

# =========================
# Style setup (transparent GraphPad-like mirror plot style)
# =========================
FONT_FAMILY = "Arial"
FONT_BOLD = True
WEIGHT = "bold" if FONT_BOLD else "normal"

rcParams["font.family"] = FONT_FAMILY
rcParams["font.weight"] = WEIGHT
rcParams["axes.labelweight"] = WEIGHT
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["axes.linewidth"] = 1.8
rcParams["figure.facecolor"] = "none"
rcParams["axes.facecolor"] = "none"
rcParams["savefig.facecolor"] = "none"
rcParams["savefig.edgecolor"] = "none"

sns.set_theme(style="white", context="talk")
sns.set_style("ticks")  # 强制显示 tick

INPUT_FILE = "Sequence/flowdata/3_REPEART.csv"
OUTPUT_DIR = Path("Sequence/flowdata/group_plots.3_styled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IS_EXCEL = False

POINT_SIZE = 8
POINT_ALPHA = 0.95
ERRORBAR_STYLE = "sd"

# mirror-plot style constants
GFP_COLOR = "#8FD98F"
MKATE_COLOR = "#F4A0A0"
EDGE_COLOR = "black"
ERR_COLOR = "black"
AXIS_TITLE_SIZE = 15
TICK_LABEL_SIZE = 13
XTICK_SIZE = 13
AXIS_LW = 2.2
BAR_EDGE_LW = 2.5
ZERO_LW = 2.5
ERR_LW = 2.5
CAPSIZE = 5
TICK_LEN = 4.5
TICK_W = 2.5
BAR_WIDTH = 0.66
MINOR_LEN = 4.0
MINOR_W = 2
# 刻度间隔：major 是显示数字的长刻度，minor 是中间的短刻度
GFP_MAJOR = 1000
GFP_MINOR = 500
MKATE_MAJOR = 10000
MKATE_MINOR = 5000
FIGSIZE_MIRROR = (11, 6.0)

if IS_EXCEL:
    df = pd.read_excel(INPUT_FILE)
else:
    df = pd.read_csv(INPUT_FILE)


def parse_numeric(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower() in ["n/a", "na", "nan", "", "none"]:
        return np.nan
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_percent_to_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower() in ["n/a", "na", "nan", "", "none"]:
        return np.nan
    s = s.replace("%", "").replace("％", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def fit_line_and_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return slope, intercept, r2


def sanitize_filename(s):
    s = str(s)
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        s = s.replace(ch, "_")
    return s.replace(" ", "_")


def ensure_dir(path_obj):
    path_obj.mkdir(parents=True, exist_ok=True)


def dedupe_legend(ax, title=None, bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique and l != "":
            unique[l] = h
    if ax.legend_:
        ax.legend_.remove()
    if unique:
        leg = ax.legend(
            unique.values(),
            unique.keys(),
            title=title,
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
            ncol=ncol,
            frameon=False,
        )
        plt.setp(leg.get_texts(), color="black")
        plt.setp(leg.get_title(), color="black")


def calc_error_array(sd_series, n_series, errorbar_style):
    sd = sd_series.fillna(0)
    n = n_series.fillna(0)
    if errorbar_style is None:
        return np.zeros(len(sd), dtype=float)
    if errorbar_style == "sd":
        return sd.values
    if errorbar_style in ["se", "sem"]:
        return (sd / np.sqrt(n.replace(0, np.nan))).fillna(0).values
    return sd.values


def thousands_formatter(v, pos):
    return f"{int(round(abs(v))):,}"


rename_map = {
    "GROUP Name": "group_name",
    "Sample:": "sample",
    "cells/Single Cells/Single Cells/Single Cells/GFP positive | Geometric Mean (BL1-A)": "dual_gfp",
    "cells/Single Cells/Single Cells/selection/Q2: BL1-A+ , YL2-A+ | Geometric Mean (YL2-A :: YL2-A)": "dual_mkate",
    "cells/Single Cells/Single Cells/selection/Q2: BL1-A+ , YL2-A+ | Freq. of Parent": "dual_freq",
    "mkate burden": "mkate_burden",
    "gfp burden": "gfp_burden",
    "sum": "burden_sum",
    "burden_index": "burden_index",
}

df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
df = df.dropna(axis=1, how="all").copy()

required_cols = [
    "group_name", "sample",
    "dual_gfp", "dual_mkate", "dual_freq",
    "mkate_burden", "gfp_burden", "burden_sum", "burden_index"
]
for c in required_cols:
    if c not in df.columns:
        df[c] = np.nan

df["group_name"] = df["group_name"].ffill()
df["sample"] = df["sample"].astype(str).str.strip()
df = df[~df["sample"].str.lower().isin({"", "nan", "none"})].copy()

for c in ["dual_gfp", "dual_mkate", "dual_freq", "mkate_burden", "gfp_burden", "burden_sum"]:
    df[c] = df[c].apply(parse_numeric)
df["burden_index"] = df["burden_index"].apply(parse_percent_to_number)

dual_tf_map = {
    "1M.fcs": "BLANK",
    "2M.fcs": "CREB1",
    "3M.fcs": "SP1",
    "4M.fcs": "YY1",
    "5M.fcs": "PLAGL2",
    "6M.fcs": "TFAP2",
    "7M.fcs": "SP1_TFAP2",
    "8M.fcs": "SP1_CREB1",
    "9M.fcs": "BLANK_CREB1",
    "10M.fcs": "SP1_YY1",
    "11M.fcs": "CREB1_YY1",
    "12M.fcs": "PLAGL2_TFAP2",
    "13M.fcs": "BLANK_SP1",
    "14M.fcs": "BLANK_YY1",
    "15M.fcs": "BLANK_PLAGL2",
    "WM.fcs": "WT_hACTB",
}

single_tf_map = {
    "1.fcs": "BLANK",
    "2.fcs": "CREB1",
    "3.fcs": "SP1",
    "4.fcs": "YY1",
    "5.fcs": "PLAGL2",
    "6.fcs": "TFAP2",
    "7.fcs": "SP1_TFAP2",
    "WT.fcs": "WT_hACTB",
}

mkate_ref_map = {
    "EM1.fcs": "EMPTY",
    "E1.fcs": "EMPTY",
}

control_samples = {
    "Hek Only.fcs": ("HEK_only", "control"),
    "HEK Only.fcs": ("HEK_only", "control"),
}


def parse_sample(sample_name):
    s = str(sample_name).strip()
    if s in dual_tf_map:
        return pd.Series({"TF": dual_tf_map[s], "sample_type": "dual"})
    if s in single_tf_map:
        return pd.Series({"TF": single_tf_map[s], "sample_type": "single"})
    if s in mkate_ref_map:
        return pd.Series({"TF": mkate_ref_map[s], "sample_type": "empty_ref"})
    if s in control_samples:
        tf, st = control_samples[s]
        return pd.Series({"TF": tf, "sample_type": st})
    return pd.Series({"TF": "Other", "sample_type": "other"})


parsed = df["sample"].apply(parse_sample)
df = pd.concat([df, parsed], axis=1)

tf_order = [
    "BLANK", "CREB1", "SP1", "YY1", "PLAGL2", "TFAP2", "SP1_TFAP2",
    "SP1_CREB1", "BLANK_CREB1", "SP1_YY1", "CREB1_YY1", "PLAGL2_TFAP2",
    "BLANK_SP1", "BLANK_YY1", "BLANK_PLAGL2", "WT_hACTB",
]

combined_order = tf_order
group_order = sorted(df["group_name"].dropna().astype(str).unique().tolist())
palette_colors = sns.color_palette("husl", n_colors=max(1, len(group_order)))
group_palette = dict(zip(group_order, palette_colors))


def style_graphpad_mirror_axes(ax, show_bottom=False):
    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LW)
    ax.spines["left"].set_color("black")
    if show_bottom:
        ax.spines["bottom"].set_linewidth(AXIS_LW)
        ax.spines["bottom"].set_color("black")
    else:
        ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", which="major", direction="out", length=TICK_LEN, width=TICK_W,
                   labelsize=TICK_LABEL_SIZE, colors="black", left=True)
    ax.tick_params(axis="y", which="minor", direction="out", length=MINOR_LEN, width=MINOR_W,
                   colors="black", left=True)
    ax.grid(False)


def save_all_groups_mirrored_dual_plot(df, output_dir, tf_order, group_palette):
    gfp_df = df[
        (df["sample_type"] == "dual") &
        (df["TF"].isin(tf_order))
    ][["group_name", "TF", "dual_gfp"]].copy().dropna(subset=["dual_gfp"])

    gfp_summary = (
        gfp_df.groupby("TF", as_index=False)
        .agg(gfp_mean=("dual_gfp", "mean"), gfp_sd=("dual_gfp", "std"), gfp_n=("dual_gfp", "count"))
        .dropna(subset=["gfp_mean"])
        .sort_values("gfp_mean", ascending=False)
        .reset_index(drop=True)
    )

    if gfp_summary.empty:
        print("跳过 mirrored 图：gfp_summary 为空")
        return

    ordered_tfs = gfp_summary["TF"].tolist()
    top_summary = pd.DataFrame({"TF": ordered_tfs}).merge(gfp_summary, on="TF", how="left")

    mkate_df = df[
        (df["sample_type"] == "dual") &
        (df["TF"].isin(tf_order))
    ][["group_name", "TF", "dual_mkate"]].copy().rename(columns={"dual_mkate": "mkate_value"}).dropna(subset=["mkate_value"])

    mkate_summary = (
        mkate_df.groupby("TF", as_index=False)
        .agg(mkate_mean=("mkate_value", "mean"), mkate_sd=("mkate_value", "std"), mkate_n=("mkate_value", "count"))
    )
    bottom_summary = pd.DataFrame({"TF": ordered_tfs}).merge(mkate_summary, on="TF", how="left")

    x = np.arange(len(ordered_tfs))
    gfp_mean = top_summary["gfp_mean"].values
    gfp_err = calc_error_array(top_summary["gfp_sd"], top_summary["gfp_n"], ERRORBAR_STYLE)
    mkate_mean = bottom_summary["mkate_mean"].values
    mkate_err = calc_error_array(bottom_summary["mkate_sd"], bottom_summary["mkate_n"], ERRORBAR_STYLE)

    fig, (axT, axB) = plt.subplots(
        2, 1,
        figsize=FIGSIZE_MIRROR,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.0},
    )
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0.20, right=0.82, bottom=0.20, top=0.97)

    # top GFP
    valid_gfp = np.isfinite(gfp_mean)
    axT.bar(
        x[valid_gfp], gfp_mean[valid_gfp], BAR_WIDTH,
        color=GFP_COLOR, edgecolor=EDGE_COLOR, linewidth=BAR_EDGE_LW, zorder=3
    )
    axT.errorbar(
        x[valid_gfp], gfp_mean[valid_gfp],
        yerr=[np.zeros_like(gfp_err[valid_gfp]), gfp_err[valid_gfp]],
        fmt="none", ecolor=ERR_COLOR, elinewidth=ERR_LW,
        capsize=CAPSIZE, capthick=ERR_LW, zorder=4
    )
    style_graphpad_mirror_axes(axT, show_bottom=False)
    axT.yaxis.set_major_locator(MultipleLocator(GFP_MAJOR))
    axT.yaxis.set_minor_locator(MultipleLocator(GFP_MINOR))
    axT.set_ylabel("GFP geometric mean", color="black", fontsize=AXIS_TITLE_SIZE, fontweight=WEIGHT)
    axT.yaxis.set_label_coords(-0.14, 0.5)
    axT.axhline(0, color="black", linewidth=ZERO_LW, zorder=5)
    axT.tick_params(axis="x", bottom=False, labelbottom=False)

    point_df_top = gfp_df.copy()
    point_df_top["TF"] = pd.Categorical(point_df_top["TF"], categories=ordered_tfs, ordered=True)
    point_df_top = point_df_top.sort_values("TF")
    rng = np.random.default_rng(42)
    for group_name, subdf in point_df_top.groupby("group_name"):
        xpos = subdf["TF"].cat.codes.values.astype(float)
        jitter = rng.uniform(-0.10, 0.10, size=len(subdf))
        axT.scatter(
            xpos + jitter,
            subdf["dual_gfp"].values,
            s=POINT_SIZE * 8,
            color=group_palette.get(group_name, "gray"),
            alpha=POINT_ALPHA,
            edgecolors="none",
            zorder=6,
            label=group_name,
        )

    # bottom mKate mirrored
    valid_mkate = np.isfinite(mkate_mean)
    axB.bar(
        x[valid_mkate], mkate_mean[valid_mkate], BAR_WIDTH,
        color=MKATE_COLOR, edgecolor=EDGE_COLOR, linewidth=BAR_EDGE_LW, zorder=3
    )
    axB.errorbar(
        x[valid_mkate], mkate_mean[valid_mkate],
        yerr=[np.zeros_like(mkate_err[valid_mkate]), mkate_err[valid_mkate]],
        fmt="none", ecolor=ERR_COLOR, elinewidth=ERR_LW,
        capsize=CAPSIZE, capthick=ERR_LW, zorder=4
    )
    axB.set_ylim(0, np.nanmax(mkate_mean + mkate_err) * 1.15 if np.any(valid_mkate) else 1)
    axB.invert_yaxis()
    style_graphpad_mirror_axes(axB, show_bottom=True)
    axB.yaxis.set_major_locator(MultipleLocator(MKATE_MAJOR))
    axB.yaxis.set_minor_locator(MultipleLocator(MKATE_MINOR))
    axB.set_ylabel("mKate geometric mean", color="black", fontsize=AXIS_TITLE_SIZE, fontweight=WEIGHT)
    axB.yaxis.set_label_coords(-0.14, 0.5)
    axB.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    point_df_bottom = mkate_df.copy()
    point_df_bottom["TF"] = pd.Categorical(point_df_bottom["TF"], categories=ordered_tfs, ordered=True)
    point_df_bottom = point_df_bottom.sort_values("TF")
    rng = np.random.default_rng(42)
    for group_name, subdf in point_df_bottom.groupby("group_name"):
        xpos = subdf["TF"].cat.codes.values.astype(float)
        jitter = rng.uniform(-0.10, 0.10, size=len(subdf))
        axB.scatter(
            xpos + jitter,
            subdf["mkate_value"].values,
            s=POINT_SIZE * 8,
            color=group_palette.get(group_name, "gray"),
            alpha=POINT_ALPHA,
            edgecolors="none",
            zorder=6,
            label=group_name,
        )

    axB.set_xticks(x)
    axB.set_xticklabels(ordered_tfs, rotation=45, ha="right", fontsize=XTICK_SIZE, fontweight=WEIGHT)
    axB.tick_params(axis="x", direction="out", length=TICK_LEN, width=TICK_W, colors="black")
    axB.set_xlim(-0.7, len(ordered_tfs) - 0.3)

    handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=8,
               markerfacecolor=group_palette[g], markeredgecolor='none', label=g)
        for g in group_order if g in group_palette
    ]
    if handles:
        leg = axT.legend(handles=handles, title="Cell Passage", bbox_to_anchor=(1.02, 1),
                         loc="upper left", frameon=False)
        plt.setp(leg.get_texts(), color="black")
        plt.setp(leg.get_title(), color="black")

    fig.savefig(output_dir / "Geometric_mean_fluorescence_intensity_after_48_hours_of_co-transfection_culture.png",
                dpi=300, transparent=True, bbox_inches="tight")
    fig.savefig(output_dir / "Geometric_mean_fluorescence_intensity_after_48_hours_of_co-transfection_culture.pdf",
                transparent=True, bbox_inches="tight")
    plt.close(fig)


def export_combined_wide_table(df, output_dir, combined_order):
    summary = (
        df[df["TF"].isin(combined_order)]
        .groupby(["group_name", "TF"], as_index=False)
        .agg(
            dual_gfp_mean=("dual_gfp", "mean"),
            dual_mkate_mean=("dual_mkate", "mean"),
            dual_freq_mean=("dual_freq", "mean"),
            burden_index_mean=("burden_index", "mean"),
            burden_sum_mean=("burden_sum", "mean"),
        )
    )

    wide = summary.pivot(index="TF", columns="group_name")
    wide = wide.reindex(combined_order)
    wide.columns = [f"{col_group}_{col_metric}" for col_metric, col_group in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(output_dir / "all_groups_wide_combined.csv", index=False)
    return wide


all_outdir = OUTPUT_DIR / "all_samples"
ensure_dir(all_outdir)
save_all_groups_mirrored_dual_plot(df, all_outdir, tf_order, group_palette)
export_combined_wide_table(df, OUTPUT_DIR, combined_order)
print(f"分析完成，结果保存在：{OUTPUT_DIR.resolve()}")


