# Chapter 6 — Visual analytics

## 知识提取

性能视图包括 bubble plot、hierarchical treemap、alarm analytics graph、bad-actor chart、radar chart、高密度 alarm plot 和动态 3D bars；相关性分析包括 correlation map 与工作流；洪泛视图包括 burst plot、sequence similarity matrix 和 spiral graph。

底层指标是可复现性的关键：平均/峰值 alarm rate、unique tags、alarm duration/interval、flood exposure、priority composition、top bad actors。图只是这些统计量的视图，不能让交互过滤改变分母而不记录。

## Benchmark 映射

首期将 visual analytics 定义为诊断层而非美学排行榜：

1. 图表输入必须来自统一 `AlarmEvent/AlarmEpisode`。
2. 每个聚合保存窗口、时区、激活/恢复口径和筛选条件。
3. similarity matrix 与模型评测调用同一相似度函数。
4. bad actor 排名同时给事件数、独立 activation、持续时间，避免 chattering 独占榜首。
5. 交互视图导出的 episode ID 必须可回到原始事件行。

后续可在 `ara_artifacts/` 保存报警 rationalization 的 evidence card，但当前不以截图代替数值结果。

证据：`06_visual_analytics.txt`，PDF physical pages 389–428。
