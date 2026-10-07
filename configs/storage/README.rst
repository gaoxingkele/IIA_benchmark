Dataset storage
===============

``data_storage.v1.json`` registers the physical storage root:
``F:/aicoding/IIA_Data``. Public and processed datasets are in its
``public_datasets`` directory. Experimental outputs and user-provided books
remain in the benchmark workspace.

On this Windows machine, ``data/public_datasets`` is a directory junction to
the registered physical directory. Existing frozen experiment configurations,
source citations, data checksums and queued configuration hashes remain valid.
All reads and future downloads through their logical paths reach the F drive.

``iia_benchmark.config.storage`` validates both the logical path and the
configured junction target. Other links and traversal outside the approved
data tree are rejected. On other platforms, the Windows mount is skipped and
ordinary local repository data paths remain usable.

The migration first copies and SHA256-verifies every file. Source cleanup is
allowed only after a stable-tree audit and successful junction verification.
Its complete per-file ledger stays under ``F:/aicoding/IIA_Data/.migration_2026-10-07``;
the summary and ledger digest are recorded in the configured migration report.

The 2026-10-07 relocation verified 45,790 accessible files. Six unreadable
historical ``test_tmp`` directories are preserved separately in
``D:/aicoding/IIA_benchmark/data/legacy_restricted_test_tmp_20261007``.
Their contents could not be inspected with the current Windows permissions;
the report records this exception instead of claiming the entire original
tree was verified. Dataset registrations use the F-drive mount. Future
migrations stop if any directory cannot be scanned.
