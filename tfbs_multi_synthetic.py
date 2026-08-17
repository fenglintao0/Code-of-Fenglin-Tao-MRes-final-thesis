import os
import glob
import numpy as np
import pandas as pd
from Bio import SeqIO

BASES = ["A", "C", "G", "T"]

# ========= 0. 全局配置 =========

# 相对当前脚本所在目录的 promoter FASTA 文件路径（这里改成了具体的文件）
PROMOTER_FILE = "promoter/phase1_synthetic_hACTB.fasta"

# JASPAR PFM 子目录名（相对当前脚本所在目录）
PFM_SUBDIR = "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar"

# PFM 文件扩展名（JASPAR2026 下载的是.jaspar）
PFM_EXT = "*.jaspar"

# 结果输出文件夹（相对当前脚本所在目录）
OUTPUT_DIR = "5.0 data"

# 基础相对得分阈值
BASE_MIN_REL_SCORE = 0.85

# 相对得分计算方式："minmax" 或 "maxonly"
REL_SCORE_MODE = "maxonly"

# 对长 motif 的加分权重（L 为 motif 长度）
# effective_rel_score = rel * (1 + LENGTH_WEIGHT * log10(L))
LENGTH_WEIGHT = 0.25   # 可调，0 表示不加分


# ========= 1. 工具函数 =========

def get_base_dir():
    """返回当前脚本所在目录的绝对路径。"""
    return os.path.dirname(os.path.abspath(__file__))


def read_jaspar_pfm(pfm_path):
    """读取单个 JASPAR PFM (.jaspar)"""
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
    """将频率矩阵转为 log-odds 矩阵。"""
    if bg is None:
        bg = {b: 0.25 for b in BASES}

    mat = np.array([pwm[b] for b in BASES])  # 4 x L
    mat = (mat + 1e-6) / (mat.sum(axis=0) + 4e-6)
    bg_vec = np.array([[bg[b]] for b in BASES])
    log_odds = np.log2(mat / bg_vec)
    return log_odds  # 4 x L


def score_window(seq_window, log_odds):
    """计算一个长度 L 的窗口在给定 log-odds 矩阵上的 raw 分数。"""
    L = log_odds.shape[1]
    assert len(seq_window) == L
    score = 0.0
    for i, base in enumerate(seq_window):
        if base in BASES:
            row = BASES.index(base)
            score += log_odds[row, i]
        else:
            score += -10  # 非标准碱基给极低分
    return score


def scan_sequence(sequence, log_odds, motif_len,
                  base_min_rel_score=0.85, mode="minmax",
                  length_weight=0.0):
    """在序列上滑动窗口，返回命中结果。"""
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


# ========= 2. 对单个 FASTA 运行扫描（支持多序列） =========

def scan_fasta_file(fasta_path, pfm_files):
    """
    对一个包含多条序列的 FASTA 文件进行扫描，返回包含所有序列命中结果的 DataFrame。
    """
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
            
            # 扫描正链
            hits_plus = scan_sequence(
                sequence, log_odds, motif_len=motif_len,
                base_min_rel_score=BASE_MIN_REL_SCORE, mode=REL_SCORE_MODE, length_weight=LENGTH_WEIGHT
            )
            for start, end, strand, raw, rel, eff_rel, site_seq in hits_plus:
                all_hits.append([
                    seq_id, tf_name, motif_id, motif_len, pfm_filename, 
                    start, end, strand, raw, rel, eff_rel, site_seq
                ])

            # 扫描负链
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


# ========= 3. 主流程：直接扫描指定的 FASTA 文件 =========

def main():
    base_dir = get_base_dir()
    print("Script dir:", base_dir)

    # 1. 加载 PFM 文件
    pfm_dir = os.path.join(base_dir, PFM_SUBDIR)
    pfm_pattern = os.path.join(pfm_dir, PFM_EXT)
    pfm_files = glob.glob(pfm_pattern)
    if not pfm_files:
        raise RuntimeError(f"No PFM (.jaspar) files found in directory: {pfm_dir}")
    print(f"Found {len(pfm_files)} PFM files")

    # 2. 验证单个 FASTA 文件是否存在
    fasta_path = os.path.join(base_dir, PROMOTER_FILE)
    if not os.path.exists(fasta_path):
        raise RuntimeError(f"FASTA file not found: {fasta_path}")
    print(f"Target FASTA file: {fasta_path}")

    # 3. 准备输出目录
    output_dir = os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 4. 执行扫描
    print("\n==========")
    print(f"Scanning FASTA file: {os.path.basename(fasta_path)}")

    df = scan_fasta_file(fasta_path, pfm_files)

    # 5. 保存结果
    fasta_base = os.path.basename(fasta_path)
    fasta_root = os.path.splitext(fasta_base)[0]
    out_name = f"{fasta_root}_tfbs_scan.csv"
    out_csv = os.path.join(output_dir, out_name)

    df.to_csv(out_csv, index=False)
    print(f"  Total hits generated: {len(df)}")
    print(f"  Results written to: {out_csv}")


if __name__ == "__main__":
    main()