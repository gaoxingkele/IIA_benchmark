# Literature and reproducibility map

## Core references

| Work | Role | Reproducibility status |
|---|---|---|
| Wang, Hu & Chen, *Intelligent Industrial Alarm Systems* (2024), DOI `10.1007/978-981-97-6516-4` | Theory spine, Ch. 1–6 | Local PDF extracted with hash |
| Downs & Vogel, *A Plant-Wide Industrial Process Control Problem* (1993), DOI `10.1016/0098-1354(93)80018-I` | TEP source | Simulator repo downloaded |
| Manca & Fay, TEP Alarm Management Dataset, DOI `10.21227/326k-qr90` | Alarm benchmark data | Authenticated raw archive acquired and CRC-audited; adapter/grouped split/reference scores pending |
| Melo et al., alarm management benchmark (IEEE Access 2021) | TEP alarm evaluation protocol | Paper evidence; dataset gated |
| CASIM, *Convolutional kernel-based classification of industrial alarm floods* (DCE 2024), DOI `10.1017/dce.2024.22` | early/open-set flood classification | Official Code Ocean DOI `10.24433/CO.4874993.v1`; B level |
| ConE-AFC (IEEE Access 2024), DOI `10.1109/ACCESS.2024.3492348` | conformal uncertainty for early classification | Official artifact DOI `10.24433/CO.5512337.v2`; B level |
| Predicting uncertainty reduction (IFAC 2025), DOI `10.1016/j.ifacol.2025.11.935` | jackknife+/bifurcation candidate | Evidence only |
| AFC-RobustBench (SSRN 2026), DOI `10.2139/ssrn.6999280` | missing/spurious/timing/delay robustness | Preprint/current candidate; not leaderboard SOTA |

CASIM source page: <https://www.cambridge.org/core/journals/data-centric-engineering/article/convolutional-kernelbased-classification-of-industrial-alarm-floods/4CCF870F136462D432578EF309B6EC97>.

## Method lineage

```text
threshold/delay/deadband ──> NOZ/dynamic threshold ──> alarm event stream
                                                        │
TE/IGTE/BN RCA <──────────── process + alarm history ────┤
                                                        │
local alignment/CHARM/next alarm ──> CASIM ──> ConE-AFC ──> RobustBench
```

近期论文只在取得官方代码、锁定版本、统一 split 和统一 metrics 后才可升级到 runnable。论文作者报告的分数不会被复制进本仓库 leaderboard。
