# External reproduction artifacts

第三方源码与数据不伪装成本仓库的自有实现。接入时必须记录工件版本、许可证、哈希、容器镜像、原论文默认参数以及任何必要 patch。

## P0 Code Ocean Capsules

| Paper | Capsule | Local ignored path | Code/Data license | State |
|---|---|---|---|---|
| CASIM | `10.24433/CO.4874993.v1` | `data/public_datasets/codeocean/casim_v1/` | MIT / CC0-1.0 | complete, verified, author default rerun complete |
| ConE-AFC | `10.24433/CO.5512337.v2` | `data/public_datasets/codeocean/cone_afc_v2/` | MIT / CC0-1.0 | complete, verified, author default run started |
| BiP-AFC | `10.24433/CO.3008979.v3` | `data/public_datasets/codeocean/bip_afc_v3/` | MIT / CC0-1.0 | complete, verified, author default queued |

The exact archive, code-tree, and data-tree hashes are frozen in
`configs/reproducibility/codeocean_capsules.v1.json`. Large archives and extracted
payloads remain ignored; do not commit them to GitHub.

## Reproduction

```powershell
python scripts/paper_exact.py check --require-local --full-hash
python scripts/paper_exact.py status
python scripts/paper_exact.py run-author --paper-id faulwasser2024_casim
python scripts/paper_exact.py summarize --paper-id faulwasser2024_casim
```

Docker is the authoritative environment. On the current Windows host, Docker is
not installed, so the first pass uses isolated Python 3.9 environments with the
exact package versions from each Capsule Dockerfile. This is recorded as a native
compatibility rerun, not as a bit-identical container rerun.
