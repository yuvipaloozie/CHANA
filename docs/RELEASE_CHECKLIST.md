# Publication release checklist

## Provenance

- [ ] Reconstruct and freeze the exact 1,629/281/281 expert-real split.
- [ ] Add `scan_id` and verify no test/development scan overlap.
- [x] Resolve primary baseline/curriculum checkpoint labels by exact hash links to the supplied evaluation outputs; retain the reversed legacy filenames in provenance.
- [ ] Calculate and record SHA-256 for every released artifact.
- [ ] Record training seeds, selected epochs, and checkpoint monitors.

## Data and results

- [ ] Populate all manifests and run `validate_manifests.py --require-populated`.
- [ ] Deposit machine-readable source data for all main and supplemental figures/tables.
- [x] Inventory every panel/table in the final Word documents without committing publication image binaries.
- [ ] Deposit source CSVs that reproduce the authoritative final Word values.
- [ ] Add a source-data dictionary and units.
- [ ] Link the cleared six-TIFF sample bundle to its exact originating checkpoint.
- [ ] Verify manuscript numbers against deposited source data.

## Software

- [ ] Execute CI in a clean Python 3.10 environment.
- [ ] Smoke-test all six checkpoints with the registry.
- [ ] Replace local Drive paths with config/manifest inputs in training workflows.
- [ ] Add a one-command final evaluation entry point.
- [ ] Choose and add the institutionally approved `LICENSE`.

## Publication

- [ ] Create a tagged GitHub release.
- [ ] Archive the exact release and data/weights with a DOI.
- [ ] Update `CITATION.cff` with DOI and manuscript metadata.
- [ ] Insert final Code Availability and Data Availability statements.
- [ ] Confirm public links work without author credentials.
