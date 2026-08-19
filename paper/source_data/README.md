# Manuscript source data

PLOS requires the values underlying graphs and summary statistics to be available in reusable form. `INDEX.csv` records the files currently deposited here. The supplied full precision-recall table is gzip-compressed because the uncompressed CSV is approximately 103 MB; standard spreadsheet/data tools can read `*.csv.gz` directly or it can be expanded with `gzip -d -k`.

Currently included:

- `Figure_4A_pixel_PR_curves_full.csv.gz`: the final supplied full pixel precision-recall coordinates for all six models (compressed-file SHA-256: `ceb5239551cf27881be7eef44ec2971a7f2342b6eae5a93cee402e9692943b5b`);
- `Table_S3_paired_statistics.csv`: the finalized numerical paired comparison table supplied in the manuscript workflow.

The remaining final figure/table source files still need to be copied from the authors' definitive analysis output, rather than reconstructed from document graphics or ambiguous older asset folders.

Minimum remaining contents for the final release include:

- image-level IoU, Dice, and finite/undefined HD95 results;
- paired curriculum-minus-baseline values and corrected statistical tests;
- object precision, recall, F1, count errors, and expert-model counts;
- UMAP coordinates and Fréchet feature-distance estimates with resampling details;
- size-, density-, and foreground-fraction robustness data;
- machine-readable versions of Main Tables 1–2.

Do not use values digitized from the final figures when original analysis tables are available.
