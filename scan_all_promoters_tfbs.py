import os
import glob
import numpy as np
import pandas as pd
from Bio import SeqIO

BASES = ["A", "C", "G", "T"]

# ========= 0. 全局配置 =========

# 相对当前脚本所在目录的 promoter FASTA 文件夹
PROMOTER_DIR = "promoter"

# JASPAR PFM 子目录名（相对当前脚本所在目录）
PFM_SUBDIR = "JASPAR2026_CORE_vertebrates_non-redundant_pfms_jaspar"

# PFM 文件扩展名（JASPAR2026 下载的是.jaspar）
PFM_EXT = "*.jaspar"

# 结果输出文件夹（相对当前脚本所在目录）
OUTPUT_DIR = "2.0 data"

# 基础相对得分阈值
BASE_MIN_REL_SCORE = 0.80

# 相对得分计算方式："minmax" 或 "maxonly"
REL_SCORE_MODE = "maxonly"

# 对长 motif 的加分权重（L 为 motif 长度）
# effective_rel_score = rel * (1 + LENGTH_WEIGHT * log10(L))
LENGTH_WEIGHT = 0.25   # 可调，0 表示不加分


# ========= 1. 工具函数 =========

def get_base_dir():
    """返回当前脚本所在目录的绝对路径。"""
    return os.path.dirname(os.path.abspath(__file__))


def load_promoter_sequence(fasta_path):
    """从 FASTA 读取启动子序列（取第一条）。"""
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Promoter FASTA not found: {fasta_path}")
    record = next(SeqIO.parse(fasta_path, "fasta"))
    seq = str(record.seq).upper()
    return record, seq


def read_jaspar_pfm(pfm_path):
    """
    读取单个 JASPAR PFM (.jaspar)，兼容两种常见格式：

    1) 纯数字：
       A  2  10  0  3...
       C  5   1  9  0...
       G  3   0  1 10...
       T  0   2  0  0...

    2) 带中括号和逗号（JASPAR 2022+ 常见格式）：
       A  [2, 10, 0, 3,...]
       C  [5, 1, 9, 0,...]
       G  [3, 0, 1, 10,...]
       T  [0, 2, 0, 0,...]
    """
    pwm = {b: [] for b in BASES}
    tf_name = None
    motif_id = None

    with open(pfm_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # header 行：>MAxxxx.x  TFNAME
            if line.startswith(">"):
                parts = line[1:].split()
                motif_id = parts[0]
                tf_name = parts[1] if len(parts) > 1 else motif_id
                continue

            # 去掉行尾注释
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

            # 去除 [ ] , 等符号，只保留数字
            raw_nums = " ".join(parts[1:])
            raw_nums = (
                raw_nums.replace("[", " ").replace("]", " ").replace(",", " ")
            )
            nums = [x for x in raw_nums.split() if x]
            counts = list(map(float, nums))
            pwm[base] = counts

    # 检查四个碱基行长度是否一致
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
    """
    在序列上滑动窗口，返回所有“长度加权后的相对得分” >= base_min_rel_score 的命中。

    mode:
      - "minmax":  (raw - min_score) / (max_score - min_score)
      - "maxonly": raw / max_score

    length_weight:
      - 0: 不考虑长度，加权相当于原始 rel
      - >0: 长 motif 的 effective_rel_score 稍微放大
    """
    L = log_odds.shape[1]
    assert L == motif_len
    max_score = np.max(log_odds, axis=0).sum()
    min_score = np.min(log_odds, axis=0).sum()

    # 长度加权因子
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


# ========= 2. 对单个 FASTA 运行扫描 =========

def scan_single_promoter(base_dir, fasta_path, pfm_files):
    """
    对一个 FASTA 启动子进行扫描，返回 DataFrame。
    """
    seq_record, sequence = load_promoter_sequence(fasta_path)
    print("  Sequence ID:", seq_record.id, "length:", len(sequence))

    all_hits = []

    rc_seq = str(seq_record.seq.reverse_complement()).upper()
    N = len(sequence)

    for pfm_path in pfm_files:
        tf_name, motif_id, pwm = read_jaspar_pfm(pfm_path)
        motif_len = len(pwm["A"])
        print(f"    Scanning TF {tf_name} ({motif_id}), length={motif_len}...")

        log_odds = pwm_to_log_odds(pwm)

        hits_plus = scan_sequence(
            sequence,
            log_odds,
            motif_len=motif_len,
            base_min_rel_score=BASE_MIN_REL_SCORE,
            mode=REL_SCORE_MODE,
            length_weight=LENGTH_WEIGHT,
        )

        hits_minus_raw = scan_sequence(
            rc_seq,
            log_odds,
            motif_len=motif_len,
            base_min_rel_score=BASE_MIN_REL_SCORE,
            mode=REL_SCORE_MODE,
            length_weight=LENGTH_WEIGHT,
        )
        hits_minus = []
        for start, end, _, raw, rel, eff_rel, site_seq in hits_minus_raw:
            new_start = N - end + 1
            new_end = N - start + 1
            hits_minus.append((new_start, new_end, "-", raw, rel, eff_rel, site_seq))

        hits = hits_plus + hits_minus
        hits.sort(key=lambda x: x[0])

        print(f"      hits (EffectiveRelScore >= {BASE_MIN_REL_SCORE}): {len(hits)}")

        for start, end, strand, raw, rel, eff_rel, site_seq in hits:
            all_hits.append([
                tf_name,
                motif_id,
                motif_len,
                os.path.basename(pfm_path),
                start,
                end,
                strand,
                raw,
                rel,
                eff_rel,
                site_seq,
            ])

    df = pd.DataFrame(
        all_hits,
        columns=[
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
            by=["Start", "TF", "EffectiveRelScore"],
            ascending=[True, True, False],
        )

    return df


# ========= 3. 主流程：扫描 promoter 目录中所有 FASTA =========

def main():
    base_dir = get_base_dir()
    print("Script dir:", base_dir)

    # 准备 PFM 文件列表
    pfm_dir = os.path.join(base_dir, PFM_SUBDIR)
    pfm_pattern = os.path.join(pfm_dir, PFM_EXT)
    pfm_files = glob.glob(pfm_pattern)
    if not pfm_files:
        raise RuntimeError(
            f"No PFM (.jaspar) files found in directory: {pfm_dir}\n"
            f"Pattern used: {pfm_pattern}"
        )
    print(f"PFM directory: {pfm_dir}")
    print(f"Found {len(pfm_files)} PFM files")

    # 准备 promoter FASTA 列表
    promoter_dir = os.path.join(base_dir, PROMOTER_DIR)
    fasta_pattern = os.path.join(promoter_dir, "*.fasta")
    fasta_files = glob.glob(fasta_pattern)
    if not fasta_files:
        raise RuntimeError(f"No FASTA files found in promoter dir: {promoter_dir}")
    print(f"Promoter directory: {promoter_dir}")
    print(f"Found {len(fasta_files)} FASTA promoters")

    # 输出目录
    output_dir = os.path.join(base_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 逐个 FASTA 扫描
    for fasta_path in fasta_files:
        print("\n==========")
        print("Scanning promoter:", fasta_path)

        df = scan_single_promoter(base_dir, fasta_path, pfm_files)

        fasta_base = os.path.basename(fasta_path)
        fasta_root = os.path.splitext(fasta_base)[0]
        out_name = f"{fasta_root}_tfbs_scan.csv"
        out_csv = os.path.join(output_dir, out_name)

        df.to_csv(out_csv, index=False)
        print(f"  Total hits: {len(df)}")
        print("  Results written to:", out_csv)


if __name__ == "__main__":
    main()