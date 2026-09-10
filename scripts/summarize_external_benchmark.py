#!/usr/bin/env python3
"""Recalculate the external-method summary from per-image metrics."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "Image_Key", "Model", "IoU", "Dice", "HD95", "Count_True",
    "Count_Pred", "Count_MAE", "TP", "FP", "FN",
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def correlation(x: list[float], y: list[float]) -> float:
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x)
        * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 0.0


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    values = lambda key: [float(row[key]) for row in rows]
    totals = lambda key: sum(int(float(row[key])) for row in rows)

    tp, fp, fn = totals("TP"), totals("FP"), totals("FN")
    precision = tp / (tp + fp + 1e-6) if tp + fp else 0.0
    recall = tp / (tp + fn + 1e-6) if tp + fn else 0.0
    object_f1 = (
        2 * precision * recall / (precision + recall + 1e-6)
        if precision + recall
        else 0.0
    )
    truth, prediction = values("Count_True"), values("Count_Pred")
    pearson_r = correlation(truth, prediction)
    differences = [pred - true for pred, true in zip(prediction, truth)]
    bias = mean(differences)
    difference_sd = statistics.stdev(differences)

    return {
        "Model": rows[0]["Model"],
        "Mean_Pixel_IoU": mean(values("IoU")),
        "Median_Pixel_IoU": statistics.median(values("IoU")),
        "Mean_Pixel_Dice": mean(values("Dice")),
        "Median_HD95_px": statistics.median(values("HD95")),
        "Mean_HD95_px": mean(values("HD95")),
        "Global_Object_Precision": precision,
        "Global_Object_Recall": recall,
        "Global_Object_F1": object_f1,
        "Mean_Count_MAE": mean(values("Count_MAE")),
        "Count_R2": pearson_r ** 2,
        "Count_Pearson_r": pearson_r,
        "Bland_Altman_Bias": bias,
        "Bland_Altman_SD": difference_sd,
        "Bland_Altman_LowerLoA": bias - 1.96 * difference_sd,
        "Bland_Altman_UpperLoA": bias + 1.96 * difference_sd,
        "Total_GT_Cells": totals("Count_True"),
        "Total_Pred_Cells": totals("Count_Pred"),
        "Total_TP": tp,
        "Total_FP": fp,
        "Total_FN": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("per_image_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.per_image_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for row in reader:
            key = (row["Model"], row["Image_Key"])
            if key in seen:
                raise ValueError(f"Duplicate model/image row: {key}")
            seen.add(key)
            grouped[row["Model"]].append(row)

    summaries = [summarize(rows) for rows in grouped.values()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    counts = {model: len(rows) for model, rows in grouped.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Methods do not contain equal image counts: {counts}")
    print(f"Wrote {args.output} from {counts}")


if __name__ == "__main__":
    main()
