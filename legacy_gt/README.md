# 旧 GT 导航

这里集中保存直接使用真实物体信息的旧演示，不是视觉导航策略。

| 文件 | 用途 |
| --- | --- |
| `run.py` | 导出、行走测试、语言规划和回放入口 |
| `planning.py` | GT 目标与路线规划 |
| `session.py`、`task_graph.py` | 连续任务、暂停与恢复 |
| `server.py`、`web.html` | GT 网页后端与页面 |
| `annotations.json` | 旧场景的人工属性标注 |

在项目根目录运行：

```bash
python -m legacy_gt.run --mode smoke
python serve.py --mode gt --port 8765
```

视觉网页的 `/gt` 入口仍可使用。两套流程共用 `simulation/` 的场景加载和运动执行，以及 `app/model_settings.py` 的配置读取。独立评分与 GT 对照实验保留在 `scripts/evaluation/`。
