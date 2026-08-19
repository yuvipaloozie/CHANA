# PLOS Computational Biology repository mapping

This checklist maps the repository to common code/data expectations; the journal's current submission guidance remains authoritative.

| Expectation | Repository response | Release state |
|---|---|---|
| Code available to editors/reviewers and public at publication | Public GitHub repository plus planned DOI archive | GitHub present; archive pending |
| Clear instructions and dependencies | Root README and environment files | Scaffold complete; training smoke test pending |
| Data underlying figures and tables in reusable form | `paper/source_data/` | Pending actual CSV deposit |
| Data availability statement with persistent location | DOI-backed data/source-data archive | Pending |
| Code availability statement | GitHub URL, release tag, commit, and DOI | Pending final tag/DOI |
| Stable software citation | `CITATION.cff` | DOI/manuscript fields pending |
| Licensing | Institutionally approved LICENSE file | Pending author/institution decision |
| Reproducible analytical provenance | Manifests, configs, checkpoint hashes, expected outputs | Schemas complete; population pending |
| No test-set leakage | Frozen test manifest and documented evaluation boundary | Must be verified before release |

Final manuscript statements should cite an immutable release, not only the moving `main` branch.
