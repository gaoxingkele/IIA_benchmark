# P0 Paper-Exact 启动报告

日期：2026-08-30  
范围：CASIM、ConE-AFC、Predicting Uncertainty Reduction / BiP-AFC

## 本轮结论

三项 P0 的“官方工件缺失”阻塞已经解除。三个公开 Code Ocean Capsule 均已完整下载、解压、许可核对并按 archive/code/data 三层哈希冻结。现有 TEP/NPP/FCC 工程结果继续作为 `transfer_result`，不修改原测试集、不筛选 seed；论文原域结果进入独立的 `paper_exact_result` 路径。

CASIM 作者代码第一遍已经完成。官方 v1 默认协议的五折闭集 balanced accuracy 为：

| Fold | BA |
|---:|---:|
| 1 | 1.000000 |
| 2 | 1.000000 |
| 3 | 0.989011 |
| 4 | 0.989011 |
| 5 | 0.991071 |
| Mean | **0.993819** |

该结果支持论文“完整已知类序列接近满分”的结论，但只能记为 `P2_author_capsule_default`。原因是 v1 的 `main.py` 明确使用 `open_set=False`，没有执行论文的 14 次留一类、70 个 train-test set、0.001–1.000 阈值扫描，因此尚不能宣称 Figure 13/14 已闭合。

数据先验还识别出两个同标签完全重复轨迹对：`19_3/2_1`（label 3）与 `61_1/96_1`（label 9）。官方 sample-stratified 五折中，fold 3–5 存在四次重复组跨 train/test；所以 0.993819 必须作为“作者原协议分数”保留，后续另报 deduplicated/grouped 敏感性结果，不能用后者覆盖前者。

使用作者原样训练出的模型，仅从测试计分中排除这 4 个“训练集存在完全相同 twin”的样本后，五折均值为 `0.993498`，相对原值下降 `0.000321`。这表明重复泄漏客观存在，但不足以解释 CASIM 的近满分闭集结果；真正尚未验证的仍是 open-set 与阈值曲线。

## 工件清单

| 项目 | DOI / version | Archive SHA-256 | Code/Data | 当前状态 |
|---|---|---|---|---|
| CASIM | `10.24433/CO.4874993.v1` | `7919b08a…a79a02c7` | 31 / 312 files | 作者默认复跑完成 |
| ConE-AFC | `10.24433/CO.5512337.v2` | `198fa6ee…cf48e8a6` | 12 / 18,751 files | 作者默认复跑进行中 |
| BiP-AFC | `10.24433/CO.3008979.v3` | `3e33382e…8d25b3f` | 11 / 2,876 files | 作者默认复跑排队 |

完整哈希、容器镜像与许可证以 `configs/reproducibility/codeocean_capsules.v1.json` 为准。大数据保持在 Git 忽略目录，仅提交清单和生成结果。

## 论文协议与 Capsule 默认值的差异

| 条目 | 论文协议 | Capsule 默认 | 处理 |
|---|---|---|---|
| CASIM | 14 次 leave-one-class-out × 5 folds；阈值扫描 | 5-fold closed-set | 默认运行只作为作者工件健全性；另加完整 open-set runner |
| ConE-AFC | 10 × 5 folds；3 alpha × 3 calibration sizes | 1 × 5 folds；alpha=.05；ncal=22 | 先原样运行，再参数化完整 3×3 网格 |
| BiP-AFC | AFC train / CP calibration / RF train 分离 | CP calibration 与 RF train 索引完全重叠 | 作者复跑保留缺陷；独立实现增加 disjoint 消融 |

## 环境

Capsule 的权威执行方式是其已归档 Docker image。当前机器没有 Docker，因此本轮先用 Python 3.9.25 和 Dockerfile 中精确版本的依赖进行 Windows 原生兼容复跑。三个环境均通过版本导入检查；Docker 复跑仍作为最终 P3 环境一致性门禁。

## 下一步闭合顺序

1. 完成 ConE-AFC 作者默认五折运行，生成 coverage 与 set-size 对照。
2. 完成 BiP-AFC 作者默认 synthetic/TEP 运行，复算 Tables 1–4。
3. 在不改变官方预处理和 fold 的前提下，扩展到论文完整参数网格。
4. 对同一 fold 运行本仓库独立实现，分离数据、协议和实现差异。
5. 只有命名表格全部在容差内才从 P2 升 P3；否则保留 Fail 及分层差距。
