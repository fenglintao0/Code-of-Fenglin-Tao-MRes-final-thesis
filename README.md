# Computational analysis scripts

Scripts for the TFBS analysis and flow cytometry figures of this thesis.
Each script runs standalone; all parameters are the ALL-CAPS constants at the top of the file.

## Environment

Python 3.9+ (tested on 3.11).

```bash
pip install "numpy>=1.20" "pandas>=1.5" "matplotlib>=3.6" "seaborn>=0.13" "biopython>=1.79" "openpyxl>=3.0"
```

seaborn **0.13+ is required** — `clustering_all_promoter_tfbs.py` uses
`sns.barplot(..., legend=False)`, which does not exist in 0.12.

## Layout

Put the scripts in `Sequence/` and run them from its parent directory — the scanning
scripts resolve paths relative to the script file, the plotting scripts relative to the
working directory, and the two only agree in this arrangement.

```
project/                          <- run from here
└── Sequence/                     <- all .py files
    ├── JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar/   (from jaspar.elixir.no)
    ├── promoter/         *.fasta
    ├── flowdata/         FlowJo *.csv
    ├── 2.0 data/  5.0 data/  tfbs_analysis_figs/                (created automatically)
```

## Pipeline

| # | Script | Output |
|---|---|---|
| 1 | `scan_all_promoters_tfbs.py` | `2.0 data/<promoter>_tfbs_scan.csv` |
| 1b | `tfbs_multi_synthetic.py` | `5.0 data/<fasta>_tfbs_scan.csv` |
| 2 | `TFBS_score_analysis_enhanced.py` | filtered top-hit CSV + score/position scatter |
| 3 | `plot_promoter_tfbs.py`, `plot_for_CMVp.py` | promoter TFBS maps (PNG) |
| 4 | `clustering_all_promoter_tfbs.py` | family counts, density heatmaps, cluster summaries |
| 5 | `Phase1plot.py`, `final_plot.py` | mirrored GFP/mKate bar plots + summary CSVs |

Stage 1 is the slow step (pure-Python scan of every matrix at every position, both
strands): minutes per promoter. Everything downstream runs in seconds.

## Notes

- Headless machines: stages 2–3 call `plt.show()`, so run as `MPLBACKEND=Agg python <script>`.
- `Phase1plot.py` requests Arial; without it matplotlib warns and falls back to DejaVu Sans.
- Flow cytometry samples are mapped to TF constructs through the `dual_tf_map` /
  `single_tf_map` dictionaries — update these per experiment, or samples fall into `Other`
  and drop out of the plots (`final_plot.py` lists them in `unrecognized_samples.csv`).
- Scanning is deterministic; scatter jitter uses a fixed seed (`default_rng(42)`).
