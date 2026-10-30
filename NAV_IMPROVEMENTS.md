# 视觉导航 P1–P4：实现、运行与验证

这是基于现有系统的改进版本，不是新的 GT 导航。GT 入口、独立评测和视觉权限边界保留。

## P1：身份、检查预算和搜索推进

- 在更新记忆前，对整帧检测和旧实例做按距离排序的一对一关联，避免按检测置信度顺序抢占旧身份。仍然是几何启发式，不保证消除实例碎片或误合并。
- 保持已选择的停靠点，只有点不再满足目标表面间距条件时才重新选择。
- 每个实例的静止检查最多六次，超限后本任务暂缓该候选，记录为未解决而非目标不存在。新任务重新给予预算。
- 搜索点覆盖增加相机投影与深度遮挡检查（地面及 0.8 m 高度采样）。同位置完成四方向扫描后必须移动，不因附近仍有遮挡点重复原地扫描。
- 原有“另一把”的固定上一目标及空间别名排除保持；没有引入 GT 身份。

## P2：场景视图记忆与分阶段选图

`ViewMemory` 保存最多 48 张关键完整观测的索引，包括位置、朝向、时间和可见实例 ID。相似位置/方向、物体集合被新观测覆盖的旧索引会被替换。原始观测文件仍留作实验日志，容量限制针对检索索引，并非自动删文件。

默认 `NAV_IMAGE_POLICY=adaptive`：

- 搜索：最多五张相关实例截图，加当前完整图。
- 接近/终点：最多三张实例截图，加当前完整图。
- 关系任务：为参照物预留截图，最多补充一张检索到的历史场景图。
- VLM 返回 `observe` 或 `uncertain` 后，下一轮允许扩展至十张实例截图，防止一次筛选丢掉证据。
- `NAV_IMAGE_POLICY=legacy` 恢复原来的最多十张截图、两张历史共同观测图，供对照。
- 图片数量是上限，模型事件记录 `visual_input` 会显示实际数量、候选 ID 和历史画面 ID。

历史图只是证据，不代替终点的新观测。当前完整图尚未绘制实例框/ID；它与截图的对应关系仍可能存在歧义。

## P3：关系主动观察

VLM 请求 `observe` 时，优先根据当前目标与记忆中的参照物寻找可达的共同观察位置：

1. 从历史视图位置、目标停靠点及周围 1/1.8/2.5 m 的采样位置生成候选。
2. 排除不可达、重复检查和过近的观察位置。
3. 检查两物体在预计镜头范围内，使用几何射线排除明显遮挡，再按路径代价排序。
4. 到达后重新检测，使用新证据更新记忆。观察位置不一定是合格停靠点，不能直接确认到达。

射线可见性使用估计包围范围容差，不保证视角一定成功；参照物检测错误也会影响观察规划。找不到新视角时暂缓候选、继续有界搜索，不凭空确认关系。

## P4：异步边走边看

- Habitat 渲染、控制、路径查询和记忆写入都在原拥有线程。
- 后台最多一个检测分割任务、一个 VLM 任务；每次复制 RGB-D/位姿和 VLM 记忆快照，不共用可变仿真状态。
- 根据最近一次实测感知耗时及剩余路径估计，只在预计能于到达前处理完时启动途中采样；短路线保留终点观测，避免重复感知增加等待。估计不保证精确。
- 行走中满足位移约 0.8 m、间隔约 0.8 s 且检测线程空闲时采样。忙时跳过采样，不堆积过期队列。
- 切换到保守几何绕行路线时连续执行可达路段，不仅因路径回退就额外触发语义判断。
- 行走沿当前已规划的几何路径，单次推进上限 12 m；每步仍受 NavMesh 和停止信号约束。同步对照仍使用 `--travel-budget`（默认 3 m）。
- 稳定的新目标候选、属性/关系变化、连续漏检当前目标等事件触发后台 VLM。普通显示帧不调用 VLM。
- 后台改变意图时先停止本段，获取新观测重新校验；后台结果不能直接宣布到达。路段结束后不会仅为待处理帧额外启动一次后台判断。
- 到达、路段中断和新搜索点仍会进行静止观察；不是全程永不停顿，也不是动态障碍规避器。
- 取消后立即停止运动，等待已启动推理结束再释放任务资源，因此界面“正在停止”可能持续到网络超时。跨任务不复用未完成的推理结果。

## 启动

在项目目录执行（GPU2 留给独立启动的 Ollama）：

```bash
# 网页：新版默认异步、按阶段选图
CUDA_VISIBLE_DEVICES=0 bash start_visual.sh cloud
# 本地模型：使用已有 Ollama 服务，仍保留启动预热
CUDA_VISIBLE_DEVICES=0 bash start_visual.sh local
# 对照：同步执行与原有图片上限
CUDA_VISIBLE_DEVICES=0 NAV_EXECUTION_MODE=sync NAV_IMAGE_POLICY=legacy bash start_visual.sh cloud
```

网页仍为 8770。切换模式需要结束旧服务再启动，并新建会话；刷新网页不会重启旧 worker。现有旧进程不会因修改代码自动更新。本次开发验证不驻留网页服务。

CLI 可显式选择：

```bash
CUDA_VISIBLE_DEVICES=0 NAV_VLM_MODEL=qwen3.8-max \
  /home/caosiqi/miniconda3/envs/habitat/bin/python run_visual.py \
  --execution-mode async --image-policy adaptive --realtime \
  --instruction '走到白色椅子旁边' --max-observations 24
```

CLI 默认同步，便于保留已有基线脚本。消融模式 `candidates` / `verify` 不启用途中后台推理。

## 验证与指标

```bash
/home/caosiqi/miniconda3/envs/habitat/bin/python -m unittest discover -p 'test_*.py'
```

新增测试覆盖跨实例抢占、深度遮挡、已扫描位置推进、视图去重与检索、图片预算及扩展、共同视角筛选、后台任务上限、快照隔离、线程归属、异常传递及静止终点核实。

真实运行结果与限制见 `P1_P4_VALIDATION.md`。单条成功不能当作成功率提升。比较总耗时还应固定场景、起点、指令、模型和是否实时播放，分别记录模型耗时、感知耗时、运动与等待；并发阶段耗时有重叠，不应简单相加。

可复用的同步/异步实测对照（顺序执行，避免两个仿真争抢 GPU）：

```bash
CUDA_VISIBLE_DEVICES=0 NO_PROXY=dashscope.aliyuncs.com,127.0.0.1,localhost \
  NAV_VLM_BACKEND=openai \
  /home/caosiqi/miniconda3/envs/habitat/bin/python compare_execution.py \
  --instruction '走到白色椅子旁边' --max-observations 12 \
  --gt outputs/evaluation/apt_0_gt.json \
  --episodes outputs/evaluation/white_episode.json \
  --output outputs/my_execution_comparison
```

输出目录必须是新的。`report.json` 包含独立评测结果、输入图片数量和途中观测数；`report.md` 包含耗时表。GT 仅在两次运行之后的评测步骤读取，不发送给导航子进程。此脚本控制执行模式；如需比较图片策略，可分别指定 `--image-policy legacy` 和 `adaptive`，使用不同输出目录。
