# 辅助脚本

从项目根目录使用 `python -m scripts.<分类>.<脚本名>`，不再使用原根目录脚本路径。

| 目录 | 用途 | 示例 |
| --- | --- | --- |
| `checks/` | 实景、网页服务、关系导航和历史帧检查，部分会用 GPU 或调用模型 | `python -m scripts.checks.check_workflow --help` |
| `evaluation/` | 独立 GT 评分、批量任务和模型／执行模式对照 | `python -m scripts.evaluation.run_visual_benchmark --help` |
| `utilities/` | 模型预热、LLM 示例导出、历史标注预览 | `python -m scripts.utilities.warmup_vlm` |

`evaluation/evaluate_visual.py` 和 `evaluation/run_gt_baseline.py` 属于评测端，GT 不传入视觉导航策略。
`utilities/preview_detection_labels.py` 针对指定历史运行，使用前需修改其中的运行目录。
日常启动仍使用根目录的 `run_visual.py`、`run_visual_session.py` 和 `serve.py`。
