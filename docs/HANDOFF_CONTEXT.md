# CHANA project handoff context

## 1. Purpose of this handoff

This document transfers the substantive decisions and technical findings from the long-form CHANA manuscript, figure-development, evaluation-notebook, and GitHub-repository workflow. It is intended to let a new repository-backed Codex project continue without reconstructing the conversation from scratch.

The immediate objective is to finish a clean, auditable GitHub repository for outside readers and PLOS Computational Biology review. The figures and tables have already been manually finalized; the next phase should focus on repository provenance, remaining source-data deposits, technical consistency checks, and restrained manuscript revision.

## 2. Scientific claim and scope

CHANA evaluates whether sequential exposure to image domains of increasing target realism can help train semantic osteoclast segmentation models when expert pixel annotations are limited. Three architectures are compared under expert-real-only baseline training and sequential curriculum training:

- U-Net with a DenseNet121 encoder;
- U-Net++ with a ResNet50 encoder and four deep-supervision outputs;
- TransUNet with an EfficientNetB0 encoder, transformer bottleneck, and three outputs.

The curriculum order is diffusion-derived synthetic, copy-paste synthetic, pseudo-labeled real, and expert-labeled real. The study evaluates semantic segmentation, watershed-derived analytical objects, counts, and morphometrics. It does not establish clinical validity, independently verify multinucleation, or demonstrate actual use in drug testing.

## 3. Authoritative dataset facts

Use these values unless a verified file-level manifest demonstrates an error:

| Dataset component | Images |
|---|---:|
| Expert-real training | 1,629 |
| Expert-real validation | 281 |
| Expert-real test | 281 |
| Total expert-real image-mask pairs | 2,191 |
| Diffusion-derived curriculum domain | 3,000 |
| Copy-paste curriculum domain | 1,500 |
| Pseudo-labeled real curriculum domain | 2,058 |

The 1,910 figure occasionally appearing in drafts is the combined expert-real training and validation pool, not the total collected real dataset and not the number of training images. The test set is a random split claimed to be independent at the original-scan level. It is not a well-level split. No separate acquisition-metadata table is available; scan information must be reconstructed from stable filenames or original records.

Historical notebook exports contain fractional `train_test_split` calls using seeds 999 and 42. Those code statements do not independently reconcile with the final 1,629/281/281 counts. The definitive public release therefore requires a populated `split_manifest.csv` with stable `image_id` and `scan_id` values.

Historical Colab paths generally use `/content/drive/MyDrive/CHANA_files` as the base. The original unprocessed image folder is named `original_images`, not `raw_rgb`. There is no separate acquisition-metadata CSV available. Additional domain-testing checkpoints were described under `CHANA Final Models and Programs/extra model weights from Domain testing/`. The supplied screenshot shows these U-Net++ filenames:

- `UNetPlusPlus_ResNet50_DS_SyntheticTrained_Phase1.weights.h5`;
- `UNetPlusPlus_ResNet50_DS_SyntheticTrained_Phase2.weights.h5`;
- `UNetPlusPlus_ResNet50_DS_SyntheticTrained_Phase2.5.weights.h5`.

These path and filename records are discovery aids only; hashes and reproduced outputs must establish identity.

## 4. Training protocol extracted from supplied code

Shared settings:

- 512 x 512 RGB inputs;
- batch size 16;
- AdamW, weight decay 1e-4;
- cosine-decay learning-rate schedule;
- horizontal and vertical flips, each p=0.5;
- rotation up to +/-20 degrees, p=0.5;
- brightness/contrast, p=0.2;
- coarse dropout, up to six 24 x 24 holes, p=0.3;
- focal Tversky parameters: alpha=0.3 for false negatives, beta=0.7 for false positives, gamma=1.5;
- soft BCE with 0.1 label smoothing where noisy-label training invokes it.

Curriculum schedule:

| Phase | Domain | Images | Epochs | Initial LR |
|---|---|---:|---:|---:|
| 1 | Diffusion-derived | 3,000 | 40 | 1e-4 |
| 2 | Copy-paste | 1,500 | 80 | 5e-5 |
| 2.5 | Pseudo-labeled real | 2,058 | 50 | 2e-5 |
| 4 | Expert-real training | 1,629 | 200 | 1e-5 |

Baseline maximum schedules:

| Architecture | Epochs | Initial LR |
|---|---:|---:|
| U-Net | 400 | 1e-4 |
| U-Net++ | 500 | 1e-4 |
| TransUNet | 400 | 1e-5 |

Important correction: the supplied pseudo-labeling code uses a fixed expert-real-fine-tuned TransUNet teacher. It does not use each architecture's immediately preceding Phase 2 checkpoint as its own teacher. Draft manuscript language describing an identical teacher/student architecture or a teacher derived only from Phase 2 requires correction.

## 5. Preprocessing, inference, and object evaluation

V9 preprocessing from the supplied inference export:

1. OpenCV BGR to RGB;
2. resize tiles to 512 x 512;
3. RGB to LAB;
4. CLAHE on the L channel, clip limit 2.0 and 8 x 8 grid;
5. 15 x 15 elliptical white top-hat;
6. weighted fusion `addWeighted(L_CLAHE, 1.0, top_hat, 0.8, 0)`;
7. LAB to RGB;
8. scale to [0,1] and normalize with ImageNet mean/std.

Large fields are padded white, tiled without overlap, predicted in batches, and stitched. Architecture output selection differs: U-Net uses its single output, U-Net++ uses the final/list-last output, and TransUNet uses list-first `final` output.

Locked analysis conventions:

- binary prediction threshold strictly greater than 0.5;
- hole filling;
- distance-transform watershed;
- peak minimum distance 20 pixels;
- evaluation-area range 50 through 10,000 pixels, inclusive in the manuscript analysis script;
- centroid Hungarian matching with maximum distance 25 pixels;
- IoU, Dice, pixel average precision/PR curve, finite HD95 plus explicit undefined-failure frequency, object precision/recall/F1, count MAE/bias/agreement, and morphometry.

Watershed objects are analytical instances derived from semantic masks. They are not expert instance annotations.

## 6. Final main figures

The final Word figure document, not older six-panel notebook composites, controls panel selection.

1. **Figure 1 - CHANA workflow.** Acquisition/tiling and preprocessing; three architectures; baseline versus four-domain sequential curriculum; locked-test outputs. A poster schematic is available as design inspiration but should be redrawn cleanly.
2. **Figure 2 - Curriculum domains and feature-space gap.** Four representative images (diffusion, copy-paste, pseudo-label, expert real), fixed-feature UMAP, and Fréchet feature distance to expert-real. The count, foreground-fraction, and area histograms were moved to supplementary material.
3. **Figure 3 - U-Net++ curriculum checkpoints.** Four column-style panels for IoU, Dice, centroid-matched object F1, and absolute count error across diffusion, copy-paste, pseudo-label, and expert-real checkpoints, with uncertainty. Historical checkpoint labels must be verified before treating this as causal phase evidence.
4. **Figure 4 - Locked-test pixel/segmentation performance.** Pixel precision-recall curves for all six models; image-level IoU box-and-whisker; finite HD95 box-and-whisker with undefined failures reported separately; curriculum-minus-baseline paired delta IoU by architecture.
5. **Figure 5 - Expert-model count agreement.** Six independent expert-count versus model-count regression/identity plots: baseline and curriculum for all three architectures. Bland-Altman values and absolute count errors are reported in tables/supplement rather than expanding the main panel.
6. **Figure 6 - Biological measurement robustness and examples.** Recall by expert-object size; IoU by reference-field density; IoU versus expert foreground fraction; prespecified representative typical case; prespecified failure case; example extracted object areas.

Foreground fraction is the proportion of reference-mask pixels labeled foreground in each image. It is used as a continuous field-complexity/coverage proxy, not a biological treatment-response measure.

## 7. Final main tables and key values

There are two main tables:

- **Table 1:** pixel- and boundary-level performance for all six models on 281 test images and 2,799 reference watershed objects: IoU, Dice, pixel AP, and finite HD95 with bootstrap confidence intervals.
- **Table 2:** object and counting performance for the same six models: object precision, recall, F1, count MAE, and count bias with confidence intervals.

The finalized table document reports curriculum U-Net++ as the strongest overall configuration: IoU 0.669, Dice 0.793, finite HD95 median/summary 42.25 pixels as defined in the table, object F1 0.842, and count MAE 1.90. Always use the exact table definition and source-data value rather than mixing mean/median HD95 summaries from older scripts.

## 8. Final supplementary set

The current combined supplementary document contains six figures and three tables, although internal heading/caption numbering should be audited before submission:

- distributions of expert object count, foreground fraction, and object area;
- V9 preprocessing sequence/intensity explanation;
- Bland-Altman agreement for six models;
- interpretability/explanatory inference outputs;
- qualitative successes and failure modes;
- other finalized supplemental panels embedded in the Word document;
- Table S1 curriculum schedule/loss specification;
- Table S2 encoders, parameters, checkpoint sizes, and inference times;
- Table S3 paired baseline-versus-curriculum statistics with within-metric Holm correction.

Known table inconsistency: the current Table S1 Word document lists 1,000 diffusion images and 1,910 expert-real images. The authoritative counts are 3,000 diffusion images and 1,629 expert-real training images, with 281 validation images separately. Correct the table/caption before final submission.

## 9. Manuscript editing constraints

The manuscript is not to be overhauled. Apply revisionary editing:

- leave the introduction substantially intact; change only grammar, concision, or necessary technical accuracy;
- keep Methods as a factual "what was done" section;
- move rationale, interpretation, and justification to Results or Discussion;
- keep Introduction, Methods, and Discussion near 2.5 double-spaced pages each in Times New Roman 11 pt;
- ensure every main and supplemental figure/table is cited in an appropriate section;
- make the Discussion comparative with other osteoclast deep-learning and relevant PLOS computational-method papers;
- do not claim actual drug-testing deployment or biological treatment-effect validation;
- preserve the distinction between expert semantic masks and model-derived watershed instances;
- correct dataset/split and pseudo-teacher descriptions using the authoritative facts above.

The working title/acronym expansion remains editorially unsettled. The manuscript currently expands CHANA as "Cell Histology Automated Neural Network Analyzer," while the repository uses a descriptive title centered on sequential domain-curriculum learning. Do not silently choose a final title without author/PI approval.

## 10. Repository state

The prepared repository scaffold includes:

- corrected README and limitations;
- `CITATION.cff`, packaging metadata, environments, CI, and contribution guidance;
- reusable V9 preprocessing, model builders, tiled inference, postprocessing, and metrics;
- one-image inference CLI;
- data/split/domain/model-registry schemas;
- PLOS compliance, release, data, training, model-card, and reproducibility documentation;
- preserved historical Python exports and manuscript-analysis scripts;
- full compressed Figure 4A PR-curve source data;
- machine-readable Supplementary Table S3 statistics;
- source-data and manifest validators;
- 10 passing unit tests.

Validation already completed locally:

```text
10 tests passed
Manifest schemas validated
Source-data checksum/dimensions validated
YAML and CITATION.cff parsed
Python syntax and git diff checks passed
```

## 11. Known repository/reproducibility blockers

1. Populate the exact dataset and split manifests with stable IDs and `scan_id`.
2. Verify no scan overlap between development and test data.
3. Resolve baseline/curriculum checkpoint identity by hashing and reproducing expected outputs. Historical registries may reverse labels, and TransUNet filenames are reused.
4. Add all six verified final checkpoints, U-Net++ phase checkpoints, and pseudo-label teacher to a DOI-backed archive; do not commit large weights directly.
5. Add remaining machine-readable source data for every main/supplemental panel and both main tables.
6. Add one redistribution-cleared smoke-test image with expected outputs.
7. Add the institutionally approved software license.
8. Create a tagged GitHub release and immutable DOI archive, then update `CITATION.cff` and manuscript availability statements.
9. Refactor the final locked evaluation into one manifest-driven command; legacy scripts still contain Colab paths and repeated cell exports.

## 12. File authority hierarchy

When files conflict, use this order:

1. explicit facts in this handoff and populated verified manifests;
2. finalized `Main Tables`, `CHANA Main Figures`, and combined supplementary Word documents;
3. machine-readable source data tied to verified checkpoints;
4. current manuscript draft;
5. technical specification document;
6. final main/supplemental analysis scripts;
7. historical training/generation notebooks and older asset bundles.

Do not promote recovered data from damaged asset ZIPs into final source data without checking it against the final figures and checkpoint hashes.

## 13. Recommended continuation order

1. Start from the GitHub-backed CHANA repository and copy/merge the prepared repository snapshot.
2. Read `AGENTS.md`, this handoff, `docs/REPRODUCIBILITY.md`, and `docs/RELEASE_CHECKLIST.md`.
3. Run the validators and tests in Python 3.10.
4. Audit the six checkpoint identities and populate `model_registry.csv`.
5. Reconstruct/populate the split manifest and verify scan independence.
6. Import the definitive remaining figure/table source data; do not rerun expensive inference solely to reproduce files that already exist.
7. Refactor only the minimum needed for a manifest-driven final evaluation.
8. Apply restrained manuscript corrections and verify all figure/table citations.
9. Obtain PI approval for final title, license, public data/weights scope, and release wording.
10. Tag and archive the verified release.
