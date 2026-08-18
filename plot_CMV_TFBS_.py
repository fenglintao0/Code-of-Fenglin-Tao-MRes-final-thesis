import os
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import itertools

# ========= Configuration =========

# Promoter FASTA file
FASTA_PATH = "Sequence/promoter/CMVp_IE.fasta"

# TFBS hit table
TFBS_PATH = "Sequence/CMVp_IE_top hits TF.xlsx"  # .xlsx or .csv


# Restrict the plot to these TFs; None plots all of them
# e.g. TF_FILTER = ["TFAP2A", "TFAP2B", "TFAP2C", "KLF5", "SP3", "E2F1"]
TF_FILTER = None

# If True, + and - strand sites are offset above/below the same TF row
SEPARATE_STRANDS = True

# Output figure
OUTPUT_FIG = "Sequence/CMVp_IE_promoter_TFBS.png"

# Annotated motifs / regions (1-based coordinates)

TATA_REGION  = (569, 575)


# ========= 1. Load the promoter sequence =========

def load_promoter_length(fasta_path):
    record = next(SeqIO.parse(fasta_path, "fasta"))
    seq_len = len(record.seq)
    print(f"Loaded promoter: {record.id}, length = {seq_len} bp")
    return seq_len

promoter_len = load_promoter_length(FASTA_PATH)

# ========= 2. Load the TFBS table =========

def load_tfbs_table(tfbs_path):
    ext = os.path.splitext(tfbs_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(tfbs_path)
    else:
        df = pd.read_csv(tfbs_path)

    # Normalise column names
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

    # Default to the + strand when no Strand column is present
    if "Strand" not in df.columns:
        df["Strand"] = "+"

    return df

df = load_tfbs_table(TFBS_PATH)
print(f"Loaded TFBS hits: {len(df)}")

if TF_FILTER is not None:
    df = df[df["TF"].isin(TF_FILTER)].copy()
    print(f"After TF filter: {len(df)} hits, TFs = {sorted(df['TF'].unique())}")

if df.empty:
    raise SystemExit("No TFBS hits to plot.")

# ========= 3. Prepare the plotting data =========

unique_tfs = sorted(df["TF"].unique())
tf_to_y = {tf: i for i, tf in enumerate(unique_tfs)}

def get_y(tf, strand):
    base_y = tf_to_y[tf]
    if not SEPARATE_STRANDS:
        return base_y
    return base_y + (0.25 if strand == "+" else -0.25)

# ========= 4. Colour palette (extended Tab20) =========

TAB20_OPTIMIZED = [
    # Tab20 dark set: clearest, assigned first
    "#1f77b4",  # dark blue
    "#ff7f0e",  # dark orange
    "#2ca02c",  # dark green
    "#d62728",  # dark red
    "#9467bd",  # dark purple
    "#8c564b",  # dark brown
    "#e377c2",  # dark pink
    "#aec7e8",  # light blue
    "#ff9896",  # light red
    "#ffbb78",  # light orange
    "#7f7f7f",  # dark grey
    "#bcbd22",  # olive
    "#17becf",  # dark cyan


    # Group A: dark red through to dark blue

    "#AE2012",  # brick red
    "#CA6702",  # pumpkin orange
    "#EE9B00",  # mustard
    "#005F73",  # deep teal blue
    "#0A9396",  # deep cyan
    "#001219",  # midnight blue-black
    "#7570B3",  # muted blue-violet
    "#E7298A",  # magenta
    "#1B9E77",  # deep teal green
    
    # Group B: extra dark tones covering the purple / green / pink range
    "#6A4C93",  # royal purple
    "#386641",  # hunter green
    "#BC6C25",  # leather brown
    "#9D0208",  # blood red
    "#457B9D",  # steel blue
    "#1D3557",  # navy
    "#7209B7",  # violet
    "#F72585",  # deep magenta
    "#2A9D8F",  # jungle green
]


unique_tfs = sorted(df["TF"].unique())
tf_to_y = {tf: i for i, tf in enumerate(unique_tfs)}

# One colour per TF
tf_to_color = {}
for i, tf in enumerate(unique_tfs):
    tf_to_color[tf] = TAB20_OPTIMIZED[i % len(TAB20_OPTIMIZED)]



# ========= 5. Draw the figure =========

fig_height = max(4, 0.45 * len(unique_tfs))  # scale the height with the number of TF rows
fig, ax = plt.subplots(figsize=(20, fig_height))

for _, row in df.iterrows():
    tf = row["TF"]
    start = int(row["Start"])
    end = int(row["End"])
    strand = str(row.get("Strand", "+"))
    y = get_y(tf, strand)
    color = tf_to_color[tf]

    width = end - start + 1

    # One rectangle per binding site
    rect = mpatches.Rectangle(
        (start, y - 0.18),   # bottom-left corner (x, y)
        width,
        0.36,                # height
        facecolor=color,
        edgecolor="black",
        alpha=0.8
    )
    ax.add_patch(rect)

    # Label each site with its TF name, in the same colour as the box
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

# ========= Annotate the core promoter elements =========

def draw_vertical_region(start, end, label, color, y_pos=None):
    """
    Shade a vertical band across the full height of the axes and label it.

    y_pos: label position in data coordinates; defaults to the middle of
    the TF rows.
    """
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

    # Default label position: near the middle of the TF rows
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
        rotation=90   # set to 0 for horizontal labels
    )


draw_vertical_region(TATA_REGION[0],  TATA_REGION[1],  "TATA BOX",  "darkorange")

# Pad the y range so the topmost TF labels are not clipped by the title
ax.set_ylim(-1, len(unique_tfs))

# Coloured y-axis labels, one per TF
ax.set_yticks([tf_to_y[tf] for tf in unique_tfs])
ax.set_yticklabels([""] * len(unique_tfs))  # cleared here, the labels are drawn manually below

for tf in unique_tfs:
    y = tf_to_y[tf]
    color = tf_to_color[tf]
    ax.text(
        -0.003,                  # just left of the axis
        y,
        tf,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="center",
        color=color,
        transform=ax.get_yaxis_transform()  # x in axes coords, y in data coords
    )



title = "TFBS distribution on CMV promoter"
ax.set_title(title, fontsize=18)

handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.show()

print(f"Figure saved to: {OUTPUT_FIG}")