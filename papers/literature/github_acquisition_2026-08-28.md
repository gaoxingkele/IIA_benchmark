# GitHub literature acquisition audit - 2026-08-28

## Source

- Repository: <https://github.com/aowo-1345/IIA_benchmark.git>
- Branch: `paper-acquisition-chenjingan`
- Audited tip: `6f21bb3fbdfba429fa755819efbbb22b976af167`
- Relationship to this repository: source `main` matched local `HEAD`
  (`0a7eb016f61e02abbc5ee0073fc24210be79e51e`); literature additions were
  isolated on the acquisition branch.

## Result

The branch tracks four licensed PDFs. Three were byte-identical to files already
present locally. One new paper was imported unchanged:

| Paper | Bytes | Pages | SHA-256 | License |
| --- | ---: | ---: | --- | --- |
| Predicting Uncertainty Reduction in Online Alarm Flood Classification | 483403 | 6 | `5fffdf0f072a4f8c2af1f6ee49968022a0223dfb4aadac18e1d9ddd591ebb2a9` | CC BY-NC-ND 4.0 |

The PDF was checked with `pdfinfo`, full-document text extraction, and visual
rendering of all six pages. Its title, authors, DOI, license, pagination, figures,
tables, and Data & Code Availability section are legible and consistent.

## Evidence boundary

The source branch manifests also describe 23 restricted or unlicensed papers as
local acquisitions from `论文大全.rar`. Those PDFs and the RAR archive are ignored
and are not present in the public Git branch. This import therefore does not mark
those papers as locally available and does not copy their manifest claims.

The newly acquired paper names official data/code artifact DOI
`10.24433/CO.3008979.v3`, resolving to Code Ocean capsule `6896356/tree/v3`.
Anonymous access currently returns HTTP 403, so paper-score reproduction remains
pending that artifact and its datasets.
