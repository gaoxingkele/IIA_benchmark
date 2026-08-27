# Configuration contract

实验配置引用 `systems/`、`datasets/`、`splits/`、`models/`、`metrics/` 五类对象，并指定输出目录。路径始终相对仓库根目录；`scripts/validate_scaffold.py` 检查引用完整性。

公开 leaderboard 的 split 必须设置 `leaderboard_eligible: true` 并描述 group key、train/validation/test 列表、open-set holdout 和随机种子。当前 `synthetic_smoke` 明确为 false。
