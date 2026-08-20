#!/usr/bin/env python
"""Verify the source-data files currently deposited in the repository."""

from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "source_data"
TABLES = ROOT / "paper" / "tables"
PR_FILE = SOURCE / "Figure_4A_pixel_PR_curves_full.csv.gz"
PR_SHA256 = "ceb5239551cf27881be7eef44ec2971a7f2342b6eae5a93cee402e9692943b5b"
TABLE_S3_SHA256 = "18cc940976b5d7adb29db33f2c8aae01e34841d3d78ca7677fd38c6a9b10cc04"
FIGURE_1_SHA256 = "fe4ae9ef77de5fe33e53b07f88730584735012f483eb9e8e63188e4fe8fdb8c2"
FINAL_TABLE_SHA256 = {
    "main_table_1.csv": "32a4a37659f23ddc5a958327d3c29f8e0a39960489a5c3a6e7033b59aad6e6d5",
    "main_table_2.csv": "60675a01b9d8e063187d64547976531f59cc6c199b9546d96c1b64158fc443d4",
    "supplementary_table_s1.csv": "3a960276d328c4f00ee08f141a4d519a0e55a5cb987d9d4bf3d495124f6fcd71",
    "supplementary_table_s2.csv": "e0b7a18f0903ab36438d53544ee6f7d3839ecd11ccfc718c5005e5c5692ccce8",
    "supplementary_table_s3.csv": TABLE_S3_SHA256,
}
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
TABLE_1_COLUMNS = [
    "Architecture", "Training regime", "Nimages", "Nobjects", "Pixel IoU",
    "Pixel Dice", "Pixel AP", "Finite HD95 (px)",
]
TABLE_2_COLUMNS = [
    "Architecture", "Training regime", "Nimages", "Nobjects",
    "Object precision", "Object recall", "Object F1", "Count MAE", "Count bias",
]
TABLE_S1_COLUMNS = [
    "Phase", "Images used", "Maximum epochs", "Batch size",
    "Initial learning rate", "Principal loss",
]
TABLE_S2_COLUMNS = [
    "Architecture", "Encoder", "Input shape", "Trainable parameters",
    "Nontrainable parameters", "Checkpoint size (MB)", "Mean seconds per image",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise SystemExit(f"{path.relative_to(ROOT)} columns mismatch")
        return list(reader)


def validate_final_tables() -> None:
    table_paths = [
        (TABLES / "main_table_1.csv", TABLE_1_COLUMNS, 6),
        (TABLES / "main_table_2.csv", TABLE_2_COLUMNS, 6),
        (TABLES / "supplementary_table_s1.csv", TABLE_S1_COLUMNS, 4),
        (TABLES / "supplementary_table_s2.csv", TABLE_S2_COLUMNS, 3),
    ]
    for path, columns, expected_rows in table_paths:
        if sha256(path) != FINAL_TABLE_SHA256[path.name]:
            raise SystemExit(f"{path.relative_to(ROOT)} differs from the finalized table")
        rows = read_csv(path, columns)
        if len(rows) != expected_rows:
            raise SystemExit(
                f"{path.relative_to(ROOT)} contains {len(rows)} rows; "
                f"expected {expected_rows}"
            )
        if any(
            value.count("(") != value.count(")")
            for row in rows for value in row.values()
        ):
            raise SystemExit(f"{path.relative_to(ROOT)} contains unbalanced parentheses")



def validate_figure_1() -> None:
    path = ROOT / "paper" / "figures" / "figure_1_overview.png"
    if not path.is_file():
        raise SystemExit("paper/figures/figure_1_overview.png is missing")
    if sha256(path) != FIGURE_1_SHA256:
        raise SystemExit("Figure 1 differs from the finalized overview panel")
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("Figure 1 is not a valid PNG")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if (width, height) != (1200, 719):
        raise SystemExit(f"Figure 1 dimensions are {(width, height)}, expected (1200, 719)")


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

    table_s3 = TABLES / "supplementary_table_s3.csv"
    if sha256(table_s3) != TABLE_S3_SHA256:
        raise SystemExit("Table S3 source-data checksum mismatch")
    with table_s3.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TABLE_S3_COLUMNS:
            raise SystemExit("Table S3 source-data columns mismatch")
        stats = list(reader)
    if len(stats) != 12 or any(row["n_paired_images"] != "281" for row in stats):
        raise SystemExit("Table S3 source-data dimensions mismatch")
    validate_final_tables()
    validate_figure_1()
    print("Deposited source data, final tables, and Figure 1 validated.")


if __name__ == "__main__":
    main()
