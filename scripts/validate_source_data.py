#!/usr/bin/env python
"""Verify the source-data files currently deposited in the repository."""

from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "source_data"
PR_FILE = SOURCE / "Figure_4A_pixel_PR_curves_full.csv.gz"
PR_SHA256 = "ceb5239551cf27881be7eef44ec2971a7f2342b6eae5a93cee402e9692943b5b"
TABLE_S3_SHA256 = "18cc940976b5d7adb29db33f2c8aae01e34841d3d78ca7677fd38c6a9b10cc04"
PR_COLUMNS = [
    "Recall_UNet_Baseline", "Precision_UNet_Baseline",
    "Recall_UNet_Curriculum", "Precision_UNet_Curriculum",
    "Recall_UNetPlusPlus_Baseline", "Precision_UNetPlusPlus_Baseline",
    "Recall_UNetPlusPlus_Curriculum", "Precision_UNetPlusPlus_Curriculum",
    "Recall_TransUNet_Baseline", "Precision_TransUNet_Baseline",
    "Recall_TransUNet_Curriculum", "Precision_TransUNet_Curriculum",
]
TABLE_S3_COLUMNS = [
    "architecture", "metric", "n_paired_images", "baseline_mean", "curriculum_mean",
    "mean_paired_difference", "ci_lower", "ci_upper", "wilcoxon_statistic",
    "p_unadjusted", "p_holm_within_metric",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if sha256(PR_FILE) != PR_SHA256:
        raise SystemExit("Figure 4A PR source-data checksum mismatch")
    with gzip.open(PR_FILE, "rt", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != PR_COLUMNS:
            raise SystemExit("Figure 4A PR source-data columns mismatch")
        pr_rows = sum(1 for _ in reader)
    if pr_rows != 522_652:
        raise SystemExit(f"Figure 4A PR source-data row count is {pr_rows}, expected 522652")

    table_s3 = SOURCE / "Table_S3_paired_statistics.csv"
    if sha256(table_s3) != TABLE_S3_SHA256:
        raise SystemExit("Table S3 source-data checksum mismatch")
    with table_s3.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TABLE_S3_COLUMNS:
            raise SystemExit("Table S3 source-data columns mismatch")
        stats = list(reader)
    if len(stats) != 12 or any(row["n_paired_images"] != "281" for row in stats):
        raise SystemExit("Table S3 source-data dimensions mismatch")
    print("Source-data files validated.")


if __name__ == "__main__":
    main()
