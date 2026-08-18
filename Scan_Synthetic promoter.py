import os
import glob
import numpy as np
import pandas as pd
from Bio import SeqIO

BASES = ["A", "C", "G", "T"]

# ========= 0. Global configuration =========

# Promoter FASTA file, relative to this script's directory
PROMOTER_FILE = "promoter/phase1_synthetic_hACTB.fasta"

# Directory holding the JASPAR PFM files, relative to this script's directory
PFM_SUBDIR = "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar"

# PFM file extension (JASPAR2026 downloads use .jaspar)
PFM_EXT = "*.jaspar"

# Output directory, relative to this script's directory
OUTPUT_DIR = "5.0 data"

# Minimum effective relative score required to report a hit
BASE_MIN_REL_SCORE = 0.85

# Relative score mode: "minmax" or "maxonly"
REL_SCORE_MODE = "maxonly"

# Length bonus for long motifs (L = motif length):
# effective_rel_score = rel * (1 + LENGTH_WEIGHT * log10(L))
LENGTH_WEIGHT = 0.25   # set to 0 to disable the length bonus


# ========= 1. Helper functions =========

def get_base_dir():
    """Return the absolute path of the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))


def read_jaspar_pfm(pfm_path):
    """Read a single JASPAR PFM (.jaspar) file."""
    pwm = {b: [] for b in BASES}
    tf_name = None
    motif_id = None

    with open(pfm_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                parts = line[1:].split()
                motif_id = parts[0]
                tf_name = parts[1] if len(parts) > 1 else motif_id
                continue

            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

            parts = line.split()
            if not parts:
                continue

            base = parts[0].upper()
            if base not in pwm:
                continue

            raw_nums = " ".join(parts[1:])
            raw_nums = (
                raw_nums.replace("[", " ").replace("]", " ").replace(",", " ")
            )
            nums = [x for x in raw_nums.split() if x]
            counts = list(map(float, nums))
            pwm[base] = counts

    lengths = [len(pwm[b]) for b in BASES]
    if len(set(lengths)) != 1 or lengths[0] == 0:
        raise ValueError(
            f"{pfm_path} is not a valid JASPAR PFM "
            f"(inconsistent A/C/G/T column lengths)."
        )

    return tf_name, motif_id, pwm


def pwm_to_log_odds(pwm, bg=None):
    """Convert a position frequency matrix into a log-odds matrix."""
    if bg is None:
        bg = {b: 0.25 for b in BASES}

    mat = np.array([pwm[b] for b in BASES])  # 4 x L
    mat = (mat + 1e-6) / (mat.sum(axis=0) + 4e-6)
    bg_vec = np.array([[bg[b]] for b in BASES])
    log_odds = np.log2(mat / bg_vec)
    return log_odds  # 4 x L


def score_window(seq_window, log_odds):
    """Score one length-L window against the given log-odds matrix."""
    L = log_odds.shape[1]
    assert len(seq_window) == L
    score = 0.0
    for i, base in enumerate(seq_window):
        if base in BASES:
            row = BASES.index(base)
            score += log_odds[row, i]
        else:
            score += -10  # heavy penalty for non-standard bases
    return score


def scan_sequence(sequence, log_odds, motif_len,
                  base_min_rel_score=0.85, mode="minmax",
                  length_weight=0.0):
    """Slide a window along the sequence and return the hits above threshold."""
    L = log_odds.shape[1]
    assert L == motif_len
    max_score = np.max(log_odds, axis=0).sum()
    min_score = np.min(log_odds, axis=0).sum()

    length_factor = 1.0 + length_weight * np.log10(L)

    hits = []
    for i in range(0, len(sequence) - L + 1):
        w = sequence[i:i + L]
        raw = score_window(w, log_odds)

        if mode == "maxonly":
            rel = raw / max_score
        else:  # "minmax"
            rel = (raw - min_score) / (max_score - min_score)

        effective_rel = rel * length_factor

        if effective_rel >= base_min_rel_score:
            hits.append((i + 1, i + L, "+", raw, rel, effective_rel, w))
    return hits


# ========= 2. Scan one FASTA file (multiple sequences supported) =========

def scan_fasta_file(fasta_path, pfm_files):
    """Scan every sequence in a FASTA file and return all hits as one DataFrame."""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        print(f"  Warning: No valid sequences found in {fasta_path}")
        return pd.DataFrame()
        
    print(f"  Found {len(records)} sequences in this FASTA file.")
    all_hits = []

    for pfm_path in pfm_files:
        tf_name, motif_id, pwm = read_jaspar_pfm(pfm_path)
        motif_len = len(pwm["A"])
        pfm_filename = os.path.basename(pfm_path)
        log_odds = pwm_to_log_odds(pwm)
        
        for seq_record in records:
            seq_id = seq_record.id
            sequence = str(seq_record.seq).upper()
            rc_seq = str(seq_record.seq.reverse_complement()).upper()
            N = len(sequence)
            
            # Forward strand
            hits_plus = scan_sequence(
                sequence, log_odds, motif_len=motif_len,
                base_min_rel_score=BASE_MIN_REL_SCORE, mode=REL_SCORE_MODE, length_weight=LENGTH_WEIGHT
            )
            for start, end, strand, raw, rel, eff_rel, site_seq in hits_plus:
                all_hits.append([
                    seq_id, tf_name, motif_id, motif_len, pfm_filename, 
                    start, end, strand, raw, rel, eff_rel, site_seq
                ])

            # Reverse strand, coordinates mapped back to the forward strand
            hits_minus_raw = scan_sequence(
                rc_seq, log_odds, motif_len=motif_len,
                base_min_rel_score=BASE_MIN_REL_SCORE, mode=REL_SCORE_MODE, length_weight=LENGTH_WEIGHT
            )
            for start, end, _, raw, rel, eff_rel, site_seq in hits_minus_raw:
                new_start = N - end + 1
                new_end = N - start + 1
                all_hits.append([
                    seq_id, tf_name, motif_id, motif_len, pfm_filename, 
                    new_start, new_end, "-", raw, rel, eff_rel, site_seq
                ])

    df = pd.DataFrame(
        all_hits,
        columns=[
            "Sequence_ID", 
            "TF",
            "Motif_ID",
            "Motif_Length",
            "PFM_File",
            "Start",
            "End",
            "Strand",
            "RawScore",
            "RelScore",
            "EffectiveRelScore",
            "SiteSeq",
        ],
    )
    
    if not df.empty:
        df = df.sort_values(
            by=["Sequence_ID", "Start", "EffectiveRelScore"],
            ascending=[True, True, False],
        )

    return df


# ========= 3. Main =========

def main():
    base_dir = get_base_dir()
    print("Script dir:", base_dir)

    # Load the PFM files
    pfm_dir = os.path.join(base_dir, PFM_SUBDIR)
    pfm_pattern = os.path.join(pfm_dir, PFM_EXT)
    pfm_files = glob.glob(pfm_pattern)
    if not pfm_files:
        raise RuntimeError(f"No PFM (.jaspar) files found in directory: {pfm_dir}")
    print(f"Found {len(pfm_files)} PFM files")

    # Check the target FASTA exists
    fasta_path = os.path.join(base_dir, PROMOTER_FILE)
    if not os.path.exists(fasta_path):
        raise RuntimeError(f"FASTA file not found: {fasta_path}")
    print(f"Target FASTA file: {fasta_path}")

    # Prepare the output directory
    output_dir = os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Run the scan
    print("\n==========")
    print(f"Scanning FASTA file: {os.path.basename(fasta_path)}")

    df = scan_fasta_file(fasta_path, pfm_files)

    # Save the results
    fasta_base = os.path.basename(fasta_path)
    fasta_root = os.path.splitext(fasta_base)[0]
    out_name = f"{fasta_root}_tfbs_scan.csv"
    out_csv = os.path.join(output_dir, out_name)

    df.to_csv(out_csv, index=False)
    print(f"  Total hits generated: {len(df)}")
    print(f"  Results written to: {out_csv}")


if __name__ == "__main__":
    main()