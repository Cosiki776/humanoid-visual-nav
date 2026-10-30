# 视觉导航第一版

本版任务设定是**几何、NavMesh、机器人定位已知，物体语义和目标位置来自在线 RGB-D**。
已有 `run.py`、`web.html` 和 GT 自然语言流程保留。`serve.py` 默认提供视觉工作台，旧 GT 页面位于 `/gt`；也可用 `--mode gt` 单独启动旧服务。

## 已检查的原项目能力

| 项目 | 原来实际实现 | 本次复用或补充 |
|---|---|---|
| 场景 | `World` 加载 ReplicaCAD、冻结刚体/关节家具 | 复用加载，`--scene` 切场景 |
| 画面 | `World.frame()` 每次从角色附近挑第三人称相机位置，实时渲染 RGB；不是全景 | 单独提供角色位置、1.65 m 高的第一人称相机；四个普通透视视角扫描 |
| RGB-D/位姿 | 原来只有 RGB 传感器，无深度观测接口 | 可选共位姿 depth 传感器、相机内参、camera-to-world、原始米制深度 |
| NavMesh | 场景加载后按半径 0.3 m、高度 1.8 m、静态物体重建 | 复用；视觉流程允许最短路径查询 |
| 规划 | GT 家具目标节点＋直连路标；LLM 输出节点路线；离线 BFS | 视觉估计包围范围生成候选停靠点；NavMesh 几何规划；VLM 不输出路线 |
| 控制 | Habitat-Lab `KinematicHumanoid`、逐帧 NavMesh 检查、直线路段跟踪 | 原 `Executor.execute_segment()`；分段约 1 m 执行，停止后采样 |
| 成功 | 距 GT 目标节点 <0.1 m；没有终点视觉核实 | 新图像、同一视觉实例、到停靠点、VLM 确认共同门控；另由独立 GT 评测 |

不是动力学机器人，也没有全身连续碰撞认证。NavMesh 圆柱近似不能保证摆动手臂无碰撞。
目标相机支持独立转动/俯仰，相当于理想可转动传感器，尚未限制真实机器人头部关节。
拍摄时仅暂时移走自己的身体并立即恢复（期间不推进物理），避免自身头部遮挡相机；没有移走家具。

## 数据泄漏审计

旧路径的 `World.objects()` 读取语义 lexicon、对象配置、handle、真实位置和包围盒；
`annotations.json` 按真实 handle 提供人工颜色；`build_map()` 围绕真实家具布目标节点。
`ModelPlanner`、`NavigationSession` 历史、`serve.py` 的 targets/navigation_map、导出的 scene.json
均带 GT。**只删 objects 字段并不能清除地图节点和会话历史中的间接泄漏。**

新路径的限制：

- `HabitatIO` 只负责可信仿真适配与几何控制；其 `GeometryWorld.objects/build_map` 直接抛权限错误。
- 不加载颜色标注，不请求 semantic 传感器，不复用 GT 路标、旧会话、网页目标列表或 scene.json。
- 感知提示词来自固定统一常用词表与用户指令解析出的类别，先按类别检测，再由 VLM 判断属性。类别不得由当前场景 GT 生成。
- `v0001` 等 ID 是在线新建视觉 ID，不是 ReplicaCAD 实例 ID；估计位置仅从掩膜、深度、位姿产生。
- VLM 只收到指令、当前 RGB、带观测编号的候选截图、视觉属性/置信度/估计距离、拒绝记录和未解决问题。
- 基准运行器读取私有 `goal_ids`，但对子进程只传 scene、seed/start、instruction、预算和模型参数。
- `evaluate_visual.py` 在单独进程导出/读取 GT，导航结束后再关联视觉点云和真实实例，结果不反馈给 Agent。

这是程序接口和数据流隔离，不是对恶意 Python 代码的操作系统级沙箱。
仿真器为加载几何必然接触资产文件；自主决策模型没有文件读取工具和仿真内部对象访问能力。
若以后开放 Agent 任意代码执行，需要再做进程/文件权限隔离。

## 模型选择与依赖

本机实查：Python 3.9.19、torch 2.0.1+cu117、transformers 4.41.2，4 张 RTX 3090（各 24 GB）。
已有 `segment_anything`、Grounding DINO tiny 的 Hugging Face 缓存，及 `../weights/sam_vit_b_01ec64.pth`。
所以第一版采用 **Grounding DINO tiny + SAM ViT-B**，没有另外安装第二套视觉栈。

SAM 3 官方推荐更新的 Python/PyTorch/CUDA 环境，并有权重访问条件：
<https://github.com/facebookresearch/sam3/blob/main/README.md>。
本轮未对 SAM 3 做图片效果实验，不能宣称 DINO+SAM 的精度优于 SAM 3；选择依据是现有依赖与权重可直接运行。
DINO 接口参考：<https://huggingface.co/docs/transformers/model_doc/grounding-dino>。
本机 transformers 的 CUDA 自定义算子编译失败，因此使用其 PyTorch 实现；只在构造模型期间禁用 eager 编译探测，
不修改系统依赖，不用失败的编译产物。模型默认只从本地缓存加载，缺权重会明确失败，不会悄悄下载或模拟检测。

VLM 使用兼容 Chat Completions 的图片接口，无需微调，支持替换为 API 或本地服务。
`NAV_VLM_MODEL`/`--vlm-model` 显式指定视觉模型，`NAV_VLM_BASE_URL`、`NAV_VLM_API_KEY` 可独立配置；
未覆盖的地址和密钥复用现有 `model_config.json`/`NAV_LLM_*`。密钥不进入实验配置或日志。
本机 `qwen-vl-plus` 返回 404；现有 `qwen3.8-max` 已实测接受图片。
新 VLM 决策关闭百炼 thinking，只记录简短可观察依据，不保存内部推理。
接口失败会记录 error，不自动切换启发式并报成功。

## 启动

以下命令均在 `/home/caosiqi/project/habitat_work/humanoid_nav_demo` 执行。
渲染与 CUDA 推理需能访问 GPU/EGL；受限沙箱内可能需要批准沙箱外运行。

```bash
# 只验证 RGB-D、四视角扫描及移动后重新观测；附几何射线校准记录
/home/caosiqi/miniconda3/envs/habitat/bin/python run_visual.py --mode capture --max-observations 8

# 真实检测分割与在线记忆，不调用 VLM
/home/caosiqi/miniconda3/envs/habitat/bin/python run_visual.py --mode perception --max-observations 4

# 完整视觉闭环
/home/caosiqi/miniconda3/envs/habitat/bin/python run_visual.py \
  --mode active_verify --vlm-model qwen3.8-max \
  --instruction '走到白色椅子旁边' --max-observations 24

# 本地视觉模型服务：须事先启动支持 image_url 的兼容服务
export NAV_VLM_BASE_URL=http://127.0.0.1:8000/v1
export NAV_VLM_MODEL=你的视觉模型名称
export NAV_VLM_API_KEY=''
/home/caosiqi/miniconda3/envs/habitat/bin/python run_visual.py --instruction '走到棕色椅子旁边'

# 显式非 VLM 调试模式，仅验证几何/记忆；不能冒充自然语言 VLM 结果
/home/caosiqi/miniconda3/envs/habitat/bin/python run_visual.py --policy color --target-category chair --instruction '走到白色椅子旁边'

# 原有 GT 流程仍可启动
/home/caosiqi/miniconda3/envs/habitat/bin/python serve.py --mode gt --port 8767
/home/caosiqi/miniconda3/envs/habitat/bin/python run.py --mode smoke
```

输出默认在不纳入 Git 的 `outputs/visual_<时间>/`。可用 `--output` 指定新目录，拒绝覆盖已有实验。
每次观测保存 RGB JPEG、无损深度/内参/位姿 NPZ、实例 mask 和去背景 crop；`result.json` 包含
语义决策、执行结果、沿途关键帧 ID、发现、拒绝证据、未解决问题、视觉记忆、状态和指标。
`trajectory.json` 保存逐控制帧轨迹。失败也尽量保存结果；模型初始化前的失败会明确报错，无伪造指标。

## 记忆与闭环约定

深度是相机光轴 Z 深度（不是射线长度），相机坐标为 X 右、Y 上、-Z 前，世界 Y 上，单位米。
过滤无效/过远深度，以稳健分位数抑制背景污染，保留多视角表面点云。
实例以类别、空间距离、包围范围重叠关联，同一帧两个 mask 不合并；多视角属性按近期证据加权，旧判断可修正。
第一版不是完整实例 SLAM：遮挡严重时可能重复实例，紧邻物体可能错配，需结合实验调阈值和加入外观特征。

VLM 输出 `select / observe / explore / confirm / reject / uncertain`，只能选视觉 ID，不能生成坐标或路线。
`confirm/reject` 必须引用当前观测、当前选中且当前检测到的实例，并已到达停靠点。
没有看到目标只会留下不确定性，不会标注整个区域为空。
主动模式允许改变目标、沿途更新、换停靠角度，探索点由已知 NavMesh 几何覆盖选择。
单段控制约 1 m；几何到达不等于语义成功。预算耗尽为 `budget_exhausted`，不宣称目标不存在。

## 独立评测与消融

```bash
# 一次运行四种设置，同起点/同指令，GT 私有信息仅留在评测端
/home/caosiqi/miniconda3/envs/habitat/bin/python run_visual_benchmark.py \
  --episodes visual_episodes.example.json --output outputs/my_benchmark \
  --vlm-model qwen3.8-max --max-observations 24

# 无 GPU 的接口/几何/评测及原项目回归
/home/caosiqi/miniconda3/envs/habitat/bin/python -m unittest discover -p 'test_*.py'
```

| 模式 | 信息与行为 |
|---|---|
| `gt` | 原 GT 物体表＋人工颜色＋路标图＋原 ModelPlanner/NavigationSession；不读取任务答案选目标 |
| `candidates` | 搜索至首次选中视觉候选，此后固定目标，几何导航，不再感知或视觉确认 |
| `verify` | 首次选中后固定目标导航，终点重新拍摄、检测、VLM 判断；核实失败即结束 |
| `active_verify` | 沿途关键帧更新、重决策、拒绝错误候选、主动换角度并终点核实 |

三个视觉设置使用相同的检测器、VLM、几何导航和起点。GT 为**现有原型的信息上限参考**，
它的路标图和模型路径规划与视觉流程不同，不能把所有差异都归因于感知。
场景/模型加载不计入导航耗时；GT 路标预处理也在计时前完成。

任务文件的 `goal_ids` 是允许的**具体真实实例**，仅供评测：如果指令只限定白色椅子，两把都满足，允许列表可有两项；
若实验要求唯一实例，应提供有唯一指代且当前系统支持的任务及单个 goal_id，不能任意选一把作为答案。
`apt_0` 已有人工核对颜色为白色/棕色，没有已确认红色椅子。因此示例红色任务是负例，
单独报告不误报/未确认比例，不并入正例导航成功率；“没有找到”仍不等于证明不存在。

评测用视觉点云落入 GT AABB 的占比关联实例，要求 ≥50%，与次高候选差 ≥15%；歧义不算成功。
匹配对象必须属于任务允许实例，最终水平距离对象包围盒 0.2–1.2 m，且运行报告到达，才计为成功。
这比只看停靠点严格，但还不是三角网格级实例归属；AABB 重叠或碰撞几何差异可能造成未决。
后续应在**评测端**增加语义渲染像素归属，绝不回传 Agent。

`scores.json` 记录逐例和聚合的成功率、认错目标率、身份未决率、路径长度、观察次数、耗时；
保留 `matched_instance`、匹配占比及最终距离。错误和预算耗尽计入失败，缺失输出直接中断，不删除失败例。
模型/API 错误单独报告 error_rate，不能算作负例上的有效不误报。
少量 smoke 实验只能证明接口闭环，不能据此得出统计显著的优劣。

## 设计参考与下一步

- [NavMCP](https://arxiv.org/html/2608.30396)：借鉴意图、沿途观察、证据记忆接口；这里执行器仍为经典几何导航，不是论文 NFM。
- [OVI-MAP](https://arxiv.org/abs/2603.26541)：借鉴增量实例与多视角语义，当前是轻量点云关联，不声称复现其映射性能。
- [SG-Nav](https://arxiv.org/abs/2410.08189)：借鉴误检后再感知。
- 用户提供的 HSGM CVPR 页面本次读取返回 403，未据此核实论文细节；本版独立实现语义决策/几何规划分工。

后续顺序：扩大正负例、已人工核对颜色的场景和种子 → 改进外观关联/遮挡核实 →
扩大通用类别评测 → 完善关系核实。现在能解析目标/参照类别并提供视觉证据，但关系任务尚未完成独立成功率验证。

## 搜不到目标时停止（按用户补充要求）

“红色椅子”只是任意目标的例子，不要求场景必须含该物体，也不需要专门补一个红色场景。
搜索采用有限的几何采样集合（NavMesh 网格 0.6 m），扫描半径 1.2 m，每个位置需要四个方向的观察。
只在同一位置完成四向扫描后记录局部覆盖，不把一帧没有检测到目标当作整个区域为空；隔墙点不会顺带标为已检查。
保持当前搜索子目标，避免隔墙近点反复切换。检查过的点不再作为新搜索目标；连续两次移动失败或同一目标下
第三次重复到访同一区域后，将该点记为本轮未解决，转向其他点，不能无限重试。

- `search_complete_no_match`：本轮可达搜索点检查完，未发现符合条件的目标，停止。
- `search_exhausted_unresolved`：剩余点受阻或反复无进展，有限候选已用尽，停止并保留未解决点。
- `budget_exhausted`：总观测预算耗尽，停止，即使尚未完成覆盖也不继续绕圈。

这些结果都不会声称“场景绝对不存在目标”：遮挡、检测漏检和有限采样仍有局限。
`search_coverage.json` 保存扫描位置、对应图像观测 ID、覆盖计数和未解决点。
可单独验证真实机器人搜索停止机制，不调用检测器或 VLM（因此不能把此模式当作语义导航成功实验）：

```bash
/home/caosiqi/miniconda3/envs/habitat/bin/python run_visual.py --mode coverage --max-observations 200
```

## 运行中发现的兼容问题

- 近目标相机必须同时调整俯仰，不能只改变 yaw，否则近处椅子会离开视野。
- NavMesh 最短路径可能贴着边界，控制器提前停步会使下一段切角受阻。视觉模式使用 0.02 m 到达阈值，
  并在直接边失效时启用 `GeometricRoadmap`：按 NavMesh 0.4 m 规则采样、严格验证直连边、Dijkstra 搜索。
  该图不含家具节点，不调用旧 `build_map()`。旧 GT 控制器默认阈值仍为 0.08 m。
- 如出现百炼 TLS EOF，按已有项目经验只对目标域名绕过代理：

  ```bash
  export NO_PROXY=dashscope.aliyuncs.com,127.0.0.1,localhost
  export no_proxy="$NO_PROXY"
  ```

  这不会自动重发模型请求，也不会隐藏网络失败。若仍失败，检查凭证、地域、网络和服务可用性。
- 依赖版本记录见 `requirements-visual.txt`；每次新实验的 `models.json` 保存版本、检测器缓存 revision、
  模型名称和初始化耗时。不要把参考依赖列表直接覆盖安装进已工作的 Habitat 环境。


## 通用类别流程（2026-10-23）

`VLMPolicy.parse_instruction()` 先用严格整句语法解析常见的“去某物旁边”单目标指令（如“去棕色沙发旁边”），避免语言模型误把停靠位置当作缺少实体。其余表达调用一次同一云端模型，只提供用户文字，生成目标类别、参照类别、属性和关系。
英文类别可超出预设词表（如 microwave），不是从场景物体表选词。无明确物体目标、多步骤或操作任务被拒绝，不静默退回椅子。
统一基础词表定义在 `visual_nav/task.py`：chair/table/sofa/bed/cabinet/refrigerator/door/lamp/television/monitor/sink/plant。
每次观察按“目标＋参照＋基础词表”去重后检测；同义词归一化。当前逐类别运行 DINO，SAM 图像编码每帧一次。
多类别会增加耗时与误检；ObjectMemory 仍按类别及几何关联，同一物体跨类别误标可能留下多个假设，尚未实现跨类别统一身份。
VLM 最多接收 10 个候选截图，优先目标，为关系参照保留最多 3 个位置。类别与颜色分数均明确为假设，终点必须新证据核实。
关系目前由 VLM 根据图像判断，尚无独立的三维关系约束求解器，不能宣称所有关系可靠支持。

启动示例（原有模型配置有效时）：

```bash
/home/caosiqi/miniconda3/envs/habitat/bin/python run_visual.py --vlm-model qwen3.8-max --instruction '走到任意一个柜子旁边'
/home/caosiqi/miniconda3/envs/habitat/bin/python run_visual.py --vlm-model qwen3.8-max --instruction '走到冰箱旁边'
```

第二条仅为用法，不代表场景必有冰箱；不存在或无法核实则按有限搜索/预算退出。
`task.json` 保存本次解析和实际检测词表，`result.json` 同样保存任务信息。
`models.json` 新增 instruction_parse_seconds；导航 elapsed_seconds 不含启动时的这一轮解析，setup_seconds 包含解析及模型加载。
纯 perception 模式默认扫描基础类别；可用 `--target-category microwave` 加入额外类别。
`--policy color` 仅为调试规则策略，必须显式指定 `--target-category`，不能代替通用自然语言理解。
GT 基线逻辑保持不变，通用类别流程已接入 CLI 和视觉 Web 首页。


## 视觉 Web 工作台（2026-10-23）

```bash
/home/caosiqi/miniconda3/envs/habitat/bin/python serve.py --port 8770
```

浏览器打开 http://127.0.0.1:8770 。远程服务器需转发 8770 端口。
默认视觉页的观察图片、候选分割图、VLM 最终结构化决策、搜索覆盖和指标均来自实际运行。
可从历史下拉框载入已完成的视觉运行，明确标注“历史记录 · 非实时”；回看事件只切换 RGB 关键帧，物体记忆仍是该次运行最新状态。
卡片点击可查看图像来源、估计位置和属性；“停止”取消本次任务并保留会话位置，停止状态不宣称任务成功。

视觉网页用 run_visual_session.py 常驻子进程承载同一会话；只有首次指令或新建会话后的首次指令初始化模型和场景。
运行中显示的“本次总等待”包含初始化，完成后的“导航耗时”来自 result.json，不包含初始化与前置指令解析。
主画面现为控制器第三人称运动流，可切换第一人称观察关键帧；下方小图始终是 VLM 观察证据。没有暂停/恢复功能。
进程标准输出的 live_state 只包含视觉记忆、观察来源、事件和几何搜索进度，不包含 GT 或密钥。独立 observer_frame 消息承载第三人称 JPEG，由网页进程缓存并通过 /observer.mjpg 发布，不写入 VLM 上下文。
结果存储在 outputs/web_visual_*；额外日志为同名 .log。取消任务保存 status=cancelled 的 result.json；会话索引在 outputs/visual_session_*，不混入成功统计。

GT 页面通过 /gt 独立访问原会话与接口，启动时才初始化旧仿真；视觉进程不会接收 GT 状态、targets 或 navigation_map。
所有原 GT 状态接口在视觉服务下加 /gt 前缀，旧自动化检查可用 --mode gt 的独立服务。
图像服务仅允许 outputs 内的观测/掩膜/截图文件，拒绝路径穿越、非图像文件与越界符号链接。


### 实时运动与沙发指令修复

Web 会话进程启用运动帧回调，复用 Executor.frame_callback：仿真按 30 Hz 实时节奏执行，每 2 帧渲染一次第三人称图（目标约 15 FPS，实际受渲染与传输影响）。
等待感知或 API 时机器人站立，画面保留最近一帧，不插值、不伪造运动。
第三人称观察只用于人类观看；下一次 RGB-D 采样会重设第一人称相机及位姿，目标感知仍只使用 obs_*.jpg 和对应深度。
每次运行保存 observer_last.jpg 供历史查看；这是最后一帧，不是整段录像。旧运行未保存第三人称图时，切换第一人称证据即可查看旧记录。
CLI 不加 --observer-stream 时维持原有快速执行模式；启用后耗时包含实时运动节奏，不应与之前的快执行结果直接比较。

已修复“去棕色沙发旁边”被语言解析误判为“无明确目标”的问题：简单单实体表达整句匹配，共用词表，不读取场景元数据；未知类别、关系和复杂表达仍交给语言解析。
原失败发生在检测开始前，不是对场景不存在沙发的判断。前置错误会写 error.json 并直接显示到网页，避免把解析错误与搜不到混淆。

### “旁边”的距离约束

默认停靠目标改为距离视觉估计物体边界0.5 m，允许0.35～0.65 m。距离指机器人基座在水平面到物体边界的距离，不是相机到物体中心的距离，也不是人体外轮廓间隙。
`visual_nav/docking.py` 沿估计包围盒各边和圆角采样，NavMesh检查可达性；同时要求附近有观测点云支持，避免停在空包围盒边角。
完成必须同时满足：到达停靠点、当前估计边界距离合格、当前帧看见选定实例、VLM新证据确认。距离不合格时即使图像很清楚，也不能标记 succeeded。
网页“距目标估计边界”每次新观察更新；result.json 的 arrival_geometry 保存阈值与边界/表面距离。
视觉估计仍存在误差，不宣称完全等于真实边界。独立评测任务可显式设置 min_target_gap_m=0.35、max_target_gap_m=0.65，GT只在事后评分使用。
旧评测默认0.2～1.2 m保留以便复现历史结果，不改写旧实验分数；新近距离验证显式使用上述较严范围。

### 途中观察频率

active_verify 选定目标后，每次推进的路程预算由1 m改为默认3 m，随后重新检测并请求VLM决策。命令行 `--travel-budget 1` 可恢复旧频率，允许0.25～5 m；result.json 的 travel_budget_m 记录配置。网页会话任务使用3 m默认预算；单次命令行实验可显式设置预算。
3 m是上限：到达目标、受阻或保守路线回退时可以提前结束。每步NavMesh检查及终点新画面核实保持不变。探索阶段仍采用四方向观察与1.5 m移动预算，candidates/verify对照模式保持原执行方式。
这项调整减少目标接近途中的同步模型等待，不提供异步感知；模型推理期间仍会停下，发现途中新候选的频率也相应降低。

### 网页连续会话

`serve.py --port 8770` 的视觉页面使用 `run_visual_session.py` 常驻进程：第一条指令初始化场景、机器人和DINO/SAM，后续指令复用它们，从当前真实位置继续。物体实例、观测编号与带图像来源的记忆跨任务保留；每次指令单独计观察预算、路径长度和耗时。单次实验入口 `run_visual.py` 仍保持独立运行，便于对照评测。

网页“停止”只取消当前任务，运动执行器在控制步检查取消并站定；正在进行的模型请求/检测需要返回后才完成取消，期间不能提交新任务。“新建会话”（`POST /api/session/new`）只允许在空闲时执行，关闭旧仿真进程、清空当前位置和记忆；下一条指令初始化新场景。历史回放不会移动当前会话机器人，回放后发出指令仍从真实当前位置继续。

`outputs/visual_session_*/session.json` 保存会话ID、指令/结果索引、当前位置和视觉记忆摘要；`visual_memory.json` 保存视觉点云及观测来源；每条指令的图像、事件和结果仍位于索引指向的 `outputs/web_visual_*` 目录，结果含session_id和task_index。查看旧会话文件不需要加载GT。

上一任务的候选拒绝原因保留在其结果中，不作为新任务的永久排除条件；搜索覆盖也按新指令重新计数，因为旧帧未必检测过新的目标类别。保留记忆不等于保证理解“刚才那个”等多轮指代，当前指令解析仍以本条明确的单目标指令为主。

当前支持服务运行期间持续会话及页面刷新，不支持服务重启后自动恢复仿真。进程意外退出时明确要求新建会话，不会悄悄从起点重跑；磁盘会话记录保留用于回看。

### 有证据来源的物体关系记忆（第一版）

`visual_nav/relations.py` 在每次检测融合后，使用同一观测中实例的视觉点云边界估计 `near`（靠近）和有方向的 `above`（上方）关系。使用当前观测的边界而非历史融合中心作此次判断；明显重叠的实例假设不连边，避免把同一表面的不同类别预测误当成相邻物体。

near要求水平边界间隙不超过1 m、水平中心距不超过2.5 m、垂直间隙不超过0.5 m；above要求水平投影覆盖较小物体至少30%，上下边界间隙0.05～1 m。阈值是保守启发式，不表示物理接触、支撑、房间归属或语义真值。

每条边保存对象ID、关系、几何依据、启发式置信分数、观测编号、时间和原始RGB路径。首次为hypothesis；连续共同观测支持后为supported（重复观测，不保证独立视角）；后续共同观测的几何不再成立时为contradicted。某帧未看到一个端点不会删除或否定旧关系。被否定关系可被新证据重新支持。

VLM最多收到候选集合内部24条关系，明确标注历史/当前、假设/重复支持/已不支持。关系任务额外携带最多2张历史共同观测原图，因为掩膜截图不保留空间上下文。提示词仍要求当前图像核实，关系不能绕过终点确认条件。网页新增“物体关系记忆”，点击可看来源原图与详细记录；结果及会话visual_memory.json保存完整关系。

部署：刷新网页可见新面板；已有常驻会话进程不会热更新Python模块，在下一次主动“新建会话”后启用新关系计算，已有历史结果不会被改写。单次CLI任务自动使用新模块。

关系模块只增加对象级几何关系与决策证据，没有生成房间/功能物体组，也没有异步感知。误检、重复实例与不完整点云仍会污染关系；supported不能当作正确率保证。

搜索推进更新：同一位置的四向观察可中断后接续；附近可达采样点全部完成检查后，直接前往未检查区域，穿过已检查走廊时不重复调用检测器和 VLM。新搜索目的地按未检查采样点收益和实际路线长度排序，选定后保持到覆盖或受阻。连续两次移动失败或位移不足 5 cm 会跳过该点；连续 40 段移动仍未进入新观察位置则以未解决状态退出。观察预算仍独立生效。覆盖仅表示有限视点检查，不证明遮挡物体不存在，也不能消除错误候选引发的绕行。

上述代码需在新的 Python 服务/会话工作进程中加载；刷新网页不会更新已经运行的会话工作进程。测试使用 `python -m unittest discover`；搜索回归包含已检查走廊通行、路线距离排序、接续扫描和无进展停止，尚不能替代真实场景成功率评测。

候选核实累计六帧仍未结束时，执行器会暂时接管搜索，完成两个扫描轮次后再允许候选打断；期间仍更新视觉物体和关系记忆，但不重复向 VLM 请求动作。这样不会因为换了候选 ID 就重置核实预算。关系任务确认到达还必须给出当前可见的 `reference_id`、`reference_verified=true` 和 `reference_reason`，并有当前共同观测支持的对应关系；这些约束仍不能保证 VLM 的类别识别正确。新版 live_state/result 包含 `runtime_version=search-progress-relations-v2`，可用来确认工作进程是否加载更新。

### “另一个”固定指代与决策恢复

会话只在任务 succeeded 且 verified 后保存 previous_verified_goal（视觉实例ID、类别及当时的估计边界/位置）。新指令要求“另一个/另外一个/换一个”时，绑定同类的上一确认目标到task.excluded_goal；本条任务中不随到达位置改变。缺少参照时明确返回说明，不猜测要排除哪一个。该功能不是任意多轮指代解析。

候选筛选与执行前检查都排除固定原目标；同类实例若中心在0.45 m内且三维估计边界重叠超过较小体积的50%或边界间隙不超过0.15 m，也视为可能的原目标别名予以排除。这是视觉几何启发式，紧邻物体或不完整点云仍可能影响判断，不使用GT匹配。

已有目标时切换到不同候选，要求新候选当前可见、至少观察2次、检测置信度不低于旧目标超过0.1，并提供switch_reason；这些只是防抖条件，不代表新候选必然正确。到达核实位置后不能直接explore/uncertain丢掉目标，应confirm、按当前证据reject或observe。另一个任务中的reject必须说明类别/属性/关系不符，不能仅以“已经到达”为由继续找下一个。

结构化决策违反证据或目标切换约束时记录decision_retry，把原因反馈给VLM，保持原目标并获取新观察，非法动作不执行。连续3次失败以decision_unresolved（目标尚未核实）结束，仍受总观察预算约束；网络、模型初始化等真正运行异常保留error。终点的当前实例可见性、证据ID、停靠距离校验不放宽。

网页可显示重新观察和未核实状态。新的Python代码在下一次新建会话后生效，不重启或重置正在使用的旧会话。
