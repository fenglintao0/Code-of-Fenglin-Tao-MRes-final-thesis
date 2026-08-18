import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# =========================
# Basic settings
# =========================
INPUT_FILE = "Sequence/flowdata/1108-2026.wsp FlowJo table.csv"
OUTPUT_DIR = Path("Sequence/flowdata/mirror_mkate_vs_empty.0810.final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IS_EXCEL = False
ERRORBAR_STYLE = "sd"   # "sd", "se", "sem", None

# Colors
GFP_BAR_COLOR = "forestgreen"
MKATE_BAR_COLOR = "lightcoral"
GRID_ALPHA = 0.35
GRID_STYLE = "--"
GRID_WIDTH = 0.8

# Font sizes
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
COMPARE_TITLE_FONTSIZE = 20
COMPARE_LABEL_FONTSIZE = 16
COMPARE_TICK_FONTSIZE = 14

# =========================
# Use ordinary GMean columns for all samples
# =========================
GFP_COLUMN = "cells/Single Cells/Single Cells/Single Cells/GFP positive | Geometric Mean (BL1-A)"
MKATE_COLUMN = "cells/Single Cells/Single Cells/selection/Q2: BL1-A+ , YL2-A+ | Geometric Mean (YL2-A :: YL2-A)"

# =========================
# 1. Read data
# =========================
if IS_EXCEL:
    df = pd.read_excel(INPUT_FILE)
else:
    df = pd.read_csv(INPUT_FILE)

print("\n=== Original columns ===")
for c in df.columns:
    print(c)

# =========================
# 2. Helpers
# =========================
def parse_numeric(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower() in ["n/a", "na", "nan", "", "none"]:
        return np.nan
    s = s.replace(",", "").replace("%", "").replace("％", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


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


def rename_tf_label(tf_name):
    tf_name = str(tf_name)
    protected = {"BLANK", "WT_hACTB", "WT_CMV", "EMPTY", "Other", "HEK_only"}
    if tf_name in protected:
        return tf_name
    if tf_name.startswith("BLANK_"):
        suffix = tf_name[len("BLANK_"):]
        return f"3_{suffix}"
    if "_" not in tf_name:
        return f"6_{tf_name}"
    return tf_name


def tf_base_name(tf_name):
    tf_name = str(tf_name)
    if tf_name.startswith("6_"):
        return tf_name[2:]
    if tf_name.startswith("3_"):
        return tf_name[2:]
    return None

# =========================
# 3. Standardize columns
# =========================
rename_map = {
    "GROUP Name": "group_name",
    "Sample:": "sample",
    GFP_COLUMN: "dual_gfp",
    MKATE_COLUMN: "dual_mkate",
    "cells/Single Cells/Single Cells/selection/Q2: BL1-A+ , YL2-A+ | Freq. of Parent": "dual_freq",
    "mkate burden": "mkate_burden",
    "gfp burden": "gfp_burden",
    "sum": "burden_sum",
    "burden_index": "burden_index",
}

df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

auto_sample_candidates = [c for c in df.columns if str(c).strip().lower() in ["sample", "sample:", "unnamed: 0"]]
if "sample" not in df.columns and auto_sample_candidates:
    df = df.rename(columns={auto_sample_candidates[0]: "sample"})

print("\n=== Renamed columns ===")
for c in df.columns:
    print(c)

# =========================
# 4. Basic cleaning
# =========================
df = df.dropna(axis=1, how="all").copy()

required_cols = ["sample", "dual_gfp", "dual_mkate"]
for c in required_cols:
    if c not in df.columns:
        df[c] = np.nan

if "group_name" not in df.columns:
    df["group_name"] = "all_samples"
else:
    df["group_name"] = df["group_name"].ffill().fillna("all_samples")

df["sample"] = df["sample"].astype(str).str.strip()
invalid_sample_tokens = {"", "nan", "none"}
df = df[~df["sample"].str.lower().isin(invalid_sample_tokens)].copy()
df = df[~df["sample"].isin(["Mean", "SD"])].copy()

for c in ["dual_gfp", "dual_mkate", "dual_freq", "mkate_burden", "gfp_burden", "burden_sum", "burden_index"]:
    if c in df.columns:
        df[c] = df[c].apply(parse_numeric)

# =========================
# 5. Parse sample -> TF + sample_type
# =========================
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
    "CMV-EGFP-M.fcs": "WT_CMV",

}

mkate_ref_map = {

    "E.fcs": "EMPTY",
}

control_samples = {
    "Hek Only.fcs": ("HEK_only", "control"),
    "HEK Only.fcs": ("HEK_only", "control"),
    "Hek Only.fc": ("HEK_only", "control"),
    "Hek Only.f": ("HEK_only", "control"),
}


def parse_sample(sample_name):
    s = str(sample_name).strip()
    if s in dual_tf_map:
        return pd.Series({"TF": dual_tf_map[s], "sample_type": "dual"})
    if s in mkate_ref_map:
        return pd.Series({"TF": mkate_ref_map[s], "sample_type": "empty_ref"})
    if s in control_samples:
        tf, st = control_samples[s]
        return pd.Series({"TF": tf, "sample_type": st})
    return pd.Series({"TF": "Other", "sample_type": "other"})


parsed = df["sample"].apply(parse_sample)
df = pd.concat([df, parsed], axis=1)
df["TF"] = df["TF"].apply(rename_tf_label)
df.to_csv(OUTPUT_DIR / "parsed_input_with_annotations.csv", index=False)

print("\n=== sample_type counts ===")
print(df["sample_type"].value_counts(dropna=False))
print("\n=== TF counts ===")
print(df["TF"].value_counts(dropna=False))

unrecognized = df[df["sample_type"] == "other"].copy()
if not unrecognized.empty:
    unrecognized.to_csv(OUTPUT_DIR / "unrecognized_samples.csv", index=False)
    print("\nWarning: unrecognized samples exported to unrecognized_samples.csv")

# =========================
# 6. Order by GFP mean descending, EMPTY last
# =========================
base_tf_order = [
    "BLANK",
    "6_CREB1",
    "6_SP1",
    "6_YY1",
    "6_PLAGL2",
    "6_TFAP2",
    "SP1_TFAP2",
    "SP1_CREB1",
    "3_CREB1",
    "SP1_YY1",
    "CREB1_YY1",
    "PLAGL2_TFAP2",
    "3_SP1",
    "3_YY1",
    "3_PLAGL2",
    "WT_hACTB",

]

order_source_df = df[
    (df["sample_type"] == "dual") &
    (df["TF"].isin(base_tf_order))
][["TF", "dual_gfp"]].copy().dropna(subset=["dual_gfp"])

if order_source_df.empty:
    tf_order = base_tf_order.copy()
    print("\nWarning: no dual_gfp data for ordering, fallback to base order")
else:
    gfp_order_summary = (
        order_source_df.groupby("TF", as_index=False)
        .agg(gfp_mean_for_order=("dual_gfp", "mean"))
    )
    gfp_order_summary["TF"] = pd.Categorical(
        gfp_order_summary["TF"],
        categories=base_tf_order,
        ordered=True
    )
    gfp_order_summary = gfp_order_summary.sort_values(
        ["gfp_mean_for_order", "TF"],
        ascending=[False, True]
    ).reset_index(drop=True)
    tf_order = gfp_order_summary["TF"].astype(str).tolist()

plot_order = tf_order + ["EMPTY"]

print("\n=== Final TF order by dual_gfp mean descending ===")
for i, tf in enumerate(tf_order, start=1):
    print(f"{i}. {tf}")

# =========================
# 7. Build plotting data
# =========================
gfp_df = df[
    (df["sample_type"] == "dual") &
    (df["TF"].isin(tf_order))
][["TF", "dual_gfp"]].copy().dropna(subset=["dual_gfp"])

gfp_summary = (
    gfp_df.groupby("TF", as_index=False)
    .agg(
        gfp_mean=("dual_gfp", "mean"),
        gfp_sd=("dual_gfp", "std"),
        gfp_n=("dual_gfp", "count"),
    )
)
gfp_summary = pd.DataFrame({"TF": plot_order}).merge(gfp_summary, on="TF", how="left")

mkate_df = df[
    ((df["sample_type"] == "dual") & (df["TF"].isin(tf_order))) |
    (df["TF"] == "EMPTY")
][["TF", "dual_mkate"]].copy().dropna(subset=["dual_mkate"])

mkate_summary = (
    mkate_df.groupby("TF", as_index=False)
    .agg(
        mkate_mean=("dual_mkate", "mean"),
        mkate_sd=("dual_mkate", "std"),
        mkate_n=("dual_mkate", "count"),
    )
)
mkate_summary = pd.DataFrame({"TF": plot_order}).merge(mkate_summary, on="TF", how="left")

summary_export = pd.DataFrame({"TF": plot_order}).merge(gfp_summary, on="TF", how="left").merge(mkate_summary, on="TF", how="left", suffixes=("_gfp", "_mkate"))
summary_export.to_csv(OUTPUT_DIR / "mirror_plot_summary.csv", index=False)

# =========================
# 8. Main mirror plot (no scatter)
# =========================
x = np.arange(len(plot_order))
gfp_mean = gfp_summary["gfp_mean"].values
gfp_err = calc_error_array(gfp_summary["gfp_sd"], gfp_summary["gfp_n"], ERRORBAR_STYLE)
mkate_mean = mkate_summary["mkate_mean"].values
mkate_err = calc_error_array(mkate_summary["mkate_sd"], mkate_summary["mkate_n"], ERRORBAR_STYLE)

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(16, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0}
)

bar_width = 0.8

valid_gfp = np.isfinite(gfp_mean)
ax1.bar(
    x[valid_gfp],
    gfp_mean[valid_gfp],
    width=bar_width,
    color=GFP_BAR_COLOR,
    edgecolor="none",
    zorder=2
)
ax1.errorbar(
    x[valid_gfp],
    gfp_mean[valid_gfp],
    yerr=gfp_err[valid_gfp],
    fmt="none",
    ecolor="black",
    elinewidth=1.5,
    capsize=5,
    capthick=1.5,
    zorder=3
)
ax1.set_ylabel("GFP geometric mean", fontsize=LABEL_FONTSIZE)
ax1.set_title("Expression strength of synthetic promoters and their corresponding monitor signal", fontsize=TITLE_FONTSIZE)
ax1.spines["bottom"].set_visible(False)
ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax1.tick_params(axis="y", labelsize=TICK_FONTSIZE)
ax1.margins(x=0.01)
ax1.grid(axis="y", linestyle=GRID_STYLE, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
ax1.grid(axis="x", visible=False)

valid_mkate = np.isfinite(mkate_mean)
ax2.bar(
    x[valid_mkate],
    -mkate_mean[valid_mkate],
    width=bar_width,
    color=MKATE_BAR_COLOR,
    edgecolor="none",
    zorder=2
)
ax2.errorbar(
    x[valid_mkate],
    -mkate_mean[valid_mkate],
    yerr=mkate_err[valid_mkate],
    fmt="none",
    ecolor="black",
    elinewidth=1.5,
    capsize=5,
    capthick=1.5,
    zorder=3
)
ax2.set_ylabel("mKate geometric mean", fontsize=LABEL_FONTSIZE)
ax2.set_xlabel("TF combination", fontsize=LABEL_FONTSIZE)
ax2.spines["top"].set_visible(False)
ax2.axhline(0, color="black", linewidth=1.2, zorder=5)
ax2.margins(x=0.01)
ax2.grid(axis="y", linestyle=GRID_STYLE, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
ax2.grid(axis="x", visible=False)
ax2.tick_params(axis="x", labelsize=TICK_FONTSIZE)
ax2.tick_params(axis="y", labelsize=TICK_FONTSIZE)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{abs(int(y))}" if np.isfinite(y) else ""))
ax2.set_xticks(x)
ax2.set_xticklabels(plot_order, rotation=35, ha="right")

plt.subplots_adjust(hspace=0)
plt.savefig(OUTPUT_DIR / "all_samples_mirrored_dual_gfp_dual_mkate_vs_EMPTY.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 9. 6_xx vs 3_xx comparison plot
# =========================
base_names_6 = {tf_base_name(tf) for tf in tf_order if str(tf).startswith("6_")}
base_names_3 = {tf_base_name(tf) for tf in tf_order if str(tf).startswith("3_")}
paired_bases = sorted([b for b in (base_names_6 & base_names_3) if b is not None])
compare_order = []
for base in paired_bases:
    compare_order.extend([f"6_{base}", f"3_{base}"])

if compare_order:
    compare_gfp_df = gfp_df[gfp_df["TF"].isin(compare_order)].copy()
    compare_mkate_df = mkate_df[mkate_df["TF"].isin(compare_order)].copy()

    compare_gfp_summary = (
        compare_gfp_df.groupby("TF", as_index=False)
        .agg(
            gfp_mean=("dual_gfp", "mean"),
            gfp_sd=("dual_gfp", "std"),
            gfp_n=("dual_gfp", "count"),
        )
    )
    compare_gfp_summary = pd.DataFrame({"TF": compare_order}).merge(compare_gfp_summary, on="TF", how="left")

    compare_mkate_summary = (
        compare_mkate_df.groupby("TF", as_index=False)
        .agg(
            mkate_mean=("dual_mkate", "mean"),
            mkate_sd=("dual_mkate", "std"),
            mkate_n=("dual_mkate", "count"),
        )
    )
    compare_mkate_summary = pd.DataFrame({"TF": compare_order}).merge(compare_mkate_summary, on="TF", how="left")

    compare_export = pd.DataFrame({"TF": compare_order}).merge(compare_gfp_summary, on="TF", how="left").merge(compare_mkate_summary, on="TF", how="left", suffixes=("_gfp", "_mkate"))
    compare_export.to_csv(OUTPUT_DIR / "compare_3_vs_6_binding_sites_summary.csv", index=False)

    x2 = np.arange(len(compare_order))
    gfp_mean2 = compare_gfp_summary["gfp_mean"].values
    gfp_err2 = calc_error_array(compare_gfp_summary["gfp_sd"], compare_gfp_summary["gfp_n"], ERRORBAR_STYLE)
    mkate_mean2 = compare_mkate_summary["mkate_mean"].values
    mkate_err2 = calc_error_array(compare_mkate_summary["mkate_sd"], compare_mkate_summary["mkate_n"], ERRORBAR_STYLE)

    fig, (bx1, bx2) = plt.subplots(
        2, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0}
    )

    valid_gfp2 = np.isfinite(gfp_mean2)
    bx1.bar(
        x2[valid_gfp2],
        gfp_mean2[valid_gfp2],
        width=0.8,
        color=GFP_BAR_COLOR,
        edgecolor="none",
        zorder=2
    )
    bx1.errorbar(
        x2[valid_gfp2],
        gfp_mean2[valid_gfp2],
        yerr=gfp_err2[valid_gfp2],
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        zorder=3
    )
    bx1.set_ylabel("GFP geometric mean", fontsize=COMPARE_LABEL_FONTSIZE)
    bx1.set_title("Comparison of transcription factors with 3 binding sites and 6 binding sites", fontsize=COMPARE_TITLE_FONTSIZE)
    bx1.spines["bottom"].set_visible(False)
    bx1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    bx1.tick_params(axis="y", labelsize=COMPARE_TICK_FONTSIZE)
    bx1.grid(axis="y", linestyle=GRID_STYLE, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
    bx1.grid(axis="x", visible=False)

    valid_mkate2 = np.isfinite(mkate_mean2)
    bx2.bar(
        x2[valid_mkate2],
        -mkate_mean2[valid_mkate2],
        width=0.8,
        color=MKATE_BAR_COLOR,
        edgecolor="none",
        zorder=2
    )
    bx2.errorbar(
        x2[valid_mkate2],
        -mkate_mean2[valid_mkate2],
        yerr=mkate_err2[valid_mkate2],
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=5,
        capthick=1.5,
        zorder=3
    )
    bx2.set_ylabel("mKate geometric mean", fontsize=COMPARE_LABEL_FONTSIZE)
    bx2.set_xlabel("TF combination", fontsize=COMPARE_LABEL_FONTSIZE)
    bx2.spines["top"].set_visible(False)
    bx2.axhline(0, color="black", linewidth=1.2, zorder=5)
    bx2.grid(axis="y", linestyle=GRID_STYLE, linewidth=GRID_WIDTH, alpha=GRID_ALPHA)
    bx2.grid(axis="x", visible=False)
    bx2.tick_params(axis="x", labelsize=COMPARE_TICK_FONTSIZE)
    bx2.tick_params(axis="y", labelsize=COMPARE_TICK_FONTSIZE)
    bx2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{abs(int(y))}" if np.isfinite(y) else ""))
    bx2.set_xticks(x2)
    bx2.set_xticklabels(compare_order, rotation=35, ha="right")

    plt.subplots_adjust(hspace=0)
    plt.savefig(OUTPUT_DIR / "comparison_3_vs_6_binding_sites_mirror.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("\nNo complete 6_xx and 3_xx TF pairs found for comparison plot.")

print(f"\nAnalysis complete. Results saved to: {OUTPUT_DIR.resolve()}")