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
INDEX_COLUMNS = ["file", "bytes", "sha256", "manuscript_item", "content"]


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


def validate_final_asset_indexes() -> None:
    assets = ROOT / "paper" / "final_assets"
    table_paths = [
        (assets / "reference_transcriptions" / "Main_Table_1_from_final_Word.csv", TABLE_1_COLUMNS, 6),
        (assets / "reference_transcriptions" / "Main_Table_2_from_final_Word.csv", TABLE_2_COLUMNS, 6),
        (assets / "reference_transcriptions" / "Supplementary_Table_S1_from_final_Word.csv", TABLE_S1_COLUMNS, 4),
        (assets / "reference_transcriptions" / "Supplementary_Table_S2_from_final_Word.csv", TABLE_S2_COLUMNS, 3),
    ]
    for path, columns, expected_rows in table_paths:
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

    for index_name in ["figure_panel_manifest.csv", "table_manifest.csv"]:
        with (assets / index_name).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            source = row.get("source_data", "").strip()
            if source and not (ROOT / source).exists():
                raise SystemExit(f"{index_name} points to missing source_data: {source}")


def validate_source_index() -> None:
    rows = read_csv(SOURCE / "INDEX.csv", INDEX_COLUMNS)
    if not rows:
        raise SystemExit("paper/source_data/INDEX.csv is empty")
    for row in rows:
        path = SOURCE / row["file"]
        if not path.is_file():
            raise SystemExit(f"INDEX.csv points to missing file: {row['file']}")
        if path.stat().st_size != int(row["bytes"]):
            raise SystemExit(f"INDEX.csv byte size mismatch: {row['file']}")
        if sha256(path) != row["sha256"]:
            raise SystemExit(f"INDEX.csv SHA-256 mismatch: {row['file']}")


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
    validate_source_index()
    validate_final_asset_indexes()
    print("Source-data files and final-asset indexes validated.")


if __name__ == "__main__":
    main()
