# 目录与入口

下列命令均在项目根目录 `humanoid_nav_demo/` 执行。

```text
run_visual.py / run_visual_session.py   单次导航 / 连续会话
serve.py / start_visual.sh              网页与启动入口
visual_nav/                            视觉导航实现
simulation/                            Habitat 仿真执行
legacy_gt/                                旧 旧 GT 入口、规划、会话及网页
app/                                   网页后端与模型配置读取
web/                                   HTML 页面
config/                                模型配置、标注和评测配置示例
requirements/                          依赖清单
tests/                                 离线测试
scripts/checks/                        实景、服务与历史帧检查
scripts/evaluation/                    评测、基线及对照实验
scripts/utilities/                     模型预热、示例导出和预览
docs/                                  设计说明与验证记录
outputs/                               运行结果（忽略提交）
```

运行离线测试：`python -m unittest discover -s tests -t .`。
辅助脚本使用模块方式，例如 `python -m scripts.evaluation.run_visual_benchmark --help`。
根目录只保留常用启动入口、依赖初始化 `bootstrap.py` 和项目说明。
本地模型配置已迁移到 `config/model_config.json`，仍被 Git 忽略。
详见 [文档索引](docs/README.md) 和 [脚本索引](scripts/README.md)。

新版 P1–P4（身份与搜索、视图记忆、关系观察、异步执行）：见 [NAV_IMPROVEMENTS.md](docs/NAV_IMPROVEMENTS.md)。

# ReplicaCAD GT → 路线规划 → Habitat 人形行走

新增独立的 **RGB-D 视觉导航第一版**：启动、模型切换、GT 权限审计和四组评测见 [VISUAL_NAV.md](docs/VISUAL_NAV.md)。
CLI 入口为 `run_visual.py`。`serve.py` 现在默认打开视觉工作台；GT 基线位于 `/gt`，也可用 `serve.py --mode gt` 单独启动。

视觉网页：在项目目录运行 `/home/caosiqi/miniconda3/envs/habitat/bin/python serve.py --port 8770`，打开 http://127.0.0.1:8770 。支持指令执行、停止、第三人称实时运动、第一人称关键帧、候选证据和历史记录。
视觉页支持连续会话：后续指令从当前位置继续，保留视觉记忆并复用模型；“停止”保留现场，“新建会话”才重置。记录保存至 `outputs/visual_session_*/session.json`，页面刷新不影响会话；服务重启暂不自动恢复仿真状态。
下文旧流程说明仍针对 GT 基线。
实测结果与已知局限见 [VISUAL_VALIDATION.md](docs/VISUAL_VALIDATION.md)。

第一版使用静态 `apt_0`、全局 GT 和 Habitat 的 `KinematicHumanoid`。
这是运动学角色演示，不是动力学机器人平衡控制，也不是视觉导航结果。
所有新增代码、依赖和输出在本目录，复用上级目录的 ReplicaCAD 和 Habitat-Lab 源码。

## 当前模式：自然语言单目标，离线多目标保留

模型请求现使用 Chat Completions SSE 流式响应：分别接收 `reasoning_content` 和 `content`。
阿里云 Qwen 请求开启 `enable_thinking`。网页在对话下方展示接口公开返回的推理文本及最终输出，
仍通过现有状态轮询约每 0.7 秒更新，并非浏览器逐 token SSE 推送。不提供推理字段的模型会明确提示。
仅完整正常结束的最终输出会解析、校验并执行；断流、长度截断不会执行半截路线。
展示仅保留本次请求最近 60000 字符，下一次请求会清空；推理文本不写入执行日志或后续模型历史。
流式不保证更快；超时仍为网络读取超时，持续收到数据的长推理可能超过该时长。

自然语言现在直接调用路线模型，不再调用任务拆解模型。每条指令只导航到一个物体；
完成后可继续输入下一条。明显的串联指令本地拒绝，其他复合语义由同一次路线模型调用判断，
返回 unsupported，不允许只执行其中一部分。自然语言识别仍有模型误判风险。
离线“两目标 + 返回”、暂停/继续、替换任务及 LangGraph 步骤执行保持不变。

网页新增可折叠“执行过程”，展示模型请求等待时间、输入字符数、路线、校验失败和行走结果；
不展示模型内部思维链。请求 JSON 去除多余空格，但不承诺解决模型自身的超时。
修改后需要重启服务并刷新网页。`scripts/checks/check_workflow.py --agent` 现验收单目标自然语言导航。

## 多步骤任务（历史设计；目前仅离线模式启用）

浏览器服务现已使用 LangGraph 管理步骤选择、执行结果、进度推进和步骤间检查点。
以下为原自然语言多任务设计记录，当前不再开放串联/返回指令；暂停和替换仍可使用：

- “先去沙发旁边，再去最近的桌子旁边，最后回来。”
- “回到会话开始的位置。”
- 执行时输入“暂停”“继续”“停止”，或使用相应按钮。
- “改成去沙发旁边”，或勾选“替换当前任务”后发送新指令。

“回来”默认返回**当前这条指令开始时的位置**；“会话开始的位置”是服务加载角色时的位置。
多个目标逐步规划，最近目标按该步骤开始时的位置判断。最多 8 个步骤，单步骤无效路线最多重规划两次。
网页显示完整任务列表、已完成步骤和当前路线。离线测试也支持“两目标 + 返回”。
普通新指令仍排队；替换会取消旧任务及其等待队列，从停止后的实际位置开始新任务。
暂停保留当前路线和剩余步骤；停止/取消会放弃剩余步骤。模型调用期间的暂停、取消要等待当前请求返回或超时。
控制短语为明确的“暂停/暂停一下/先暂停”“继续/继续执行/恢复”“停止/停下/取消”；
“改成/换成/改去/换去”开头按替换处理，其他自然语言修改建议使用“替换”勾选框。

结构与线程约束：

```text
模型拆解步骤 → LangGraph select_step → execute 边界暂停
                                      ↓ 主线程拥有者执行 Habitat 导航
                                写回执行结果 → advance → 下一步 / 结束
```

`execute` 使用 LangGraph `interrupt_before` + `update_state(as_node="execute")`。
实际 Habitat 操作留在仿真拥有线程，避免跨线程 OpenGL 崩溃。图检查点为每条任务独立的内存存储；
行走中暂停由逐帧控制门实现。它支持本次进程内暂停/继续，不支持崩溃或进程重启后的仿真状态恢复。
完整结果仍写入 `outputs/session_*/turn_*.json`。

当前 Habitat Python 为 3.9，使用兼容的 LangGraph **0.6.11**；1.x 需要 Python 3.10+。
依赖放在 `graph_vendor/`，版本见 `requirements/graph.txt`，不修改已有 conda 环境。
以后迁移 1.x 时可升级运行环境或把规划与仿真拆成两个进程。

本轮未实现：环绕多圈、禁行区域等约束、抓取、坐下、左/右等相对位置目标。
任务拆解提示词要求对不支持的任务明确返回 unsupported，不把它简化成普通导航；
自然语言是否被完整理解仍需要结合测试评估。

测试正在运行的新服务（会实际移动机器人，`--agent` 还会调用所配置的模型）：

```bash
/home/caosiqi/miniconda3/envs/habitat/bin/python -m scripts.checks.check_workflow --url http://127.0.0.1:8767
/home/caosiqi/miniconda3/envs/habitat/bin/python -m scripts.checks.check_workflow --url http://127.0.0.1:8767 --agent
```

若模型请求报 TLS EOF，而直连阿里云正常，可仅为该域名绕过环境代理启动服务：

```bash
NO_PROXY=dashscope.aliyuncs.com,127.0.0.1,localhost no_proxy=dashscope.aliyuncs.com,127.0.0.1,localhost /home/caosiqi/miniconda3/envs/habitat/bin/python serve.py --mode gt --port 8767
```

本机浏览器访问 `http://127.0.0.1:8767`；远程使用时需转发 8767 端口。

本轮验证：19 项单元测试通过；真实仿真离线三步任务、暂停/继续、中途替换通过，返回起点误差约 0.061 m。
真实 Qwen 请求已成功拆解三步任务并执行第一步；第二步受阻后的重新规划请求超时，任务安全结束，
因此尚未通过真实模型三步任务的完整验收。不要把离线通过等同于自然语言全链路稳定。

## 运行

### 浏览器实时画面与持续会话（推荐）

打开本目录的 `config/model_config.json`，在 `api_key` 的双引号内填入自己的百炼 API Key：

```json
{
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model": "qwen3.8-max",
  "api_key": "在这里填写你的 API Key",
  "timeout_seconds": 120
}
```

默认地址是**北京地域、按量付费**的兼容接口；Key 必须与地域及计费方案对应。
新加坡地址为 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`。
其他地域、业务空间专属地址或套餐地址以控制台提供的 OpenAI 兼容 Base URL 为准：
[阿里云 Base URL 文档](https://help.aliyun.com/zh/model-studio/base-url)、
[Qwen3.8-Max 模型 ID](https://help.aliyun.com/zh/model-studio/qwen3-8-max)。

`config/model_config.json` 已加入 Git 忽略规则，不会通过本网页提供下载；
无密钥模板为 `config/model_config.example.json`。程序按脚本所在目录找配置，与终端当前目录无关。
原有 `NAV_LLM_BASE_URL`、`NAV_LLM_MODEL`、`NAV_LLM_API_KEY` 环境变量仍兼容，**环境变量优先**；
若之前设置过它们且希望使用文件配置，先执行：

```bash
unset NAV_LLM_BASE_URL NAV_LLM_MODEL NAV_LLM_API_KEY
/home/caosiqi/miniconda3/envs/habitat/bin/python serve.py --mode gt --port 8765
```

浏览器访问 **http://127.0.0.1:8765**。不配置模型也能启动，展开页面的“离线目标测试”即可选择家具并连续执行。
页面不会收集或返回 API Key，模型配置在服务启动时读取；填写或修改配置后需要重启服务。
如果旧服务仍占用 8765，先在其启动终端按 Ctrl+C；也可用 `--port 8766` 启动并访问新端口。
如果在远程服务器运行，在 VS Code 的“端口 / Ports”面板转发 **8765**，再打开转发后的本地网址；
也可以用 `ssh -L 8765:127.0.0.1:8765 用户名@服务器地址` 建立端口转发。
默认绑定本机，当前是单用户共享会话，没有账户隔离。

会话行为：

- 同一个场景与人形角色只加载一次。第一条完成后，下一条从**实际当前位置**继续，不重置机器人。
- 保存最近 40 条用户 / 助手记录，模型会收到上次目标、执行结果和位置；支持接着回答模型的澄清问题。
- 多条指令依次排队（最多等待 8 条）；刷新网页会重新取得已有会话状态。
- “停止”会取消当前任务并清空等待队列；行走时逐帧检查停止信号。模型请求中的取消在响应返回或超时后生效。
- 当前是导航任务会话，不是任意聊天助手；历史超出窗口会裁掉，进程重启不会自动恢复场景状态。
- `outputs/session_<时间>/turn_0001.json` 等记录每轮起点、终点、路线和执行结果。
- 页面默认逐张请求最新 JPEG（约 10 帧/秒，具体受网络影响），避免端口转发缓冲 MJPEG 长连接。
  MJPEG 接口仍可直接访问；仿真按 30 Hz 节奏执行，每两步生成一帧。空闲或等待模型时显示最后一帧。
- 页面显示服务端帧计数、实际坐标和上次起终点位移；镜头跟随人形角色，因此人物在画面中的位置可能变化不大。

HTTP 接口：`GET /api/status`、`GET /stream.mjpg`、`GET /api/frame.jpg`；
`POST /api/command` 接收 `{"instruction":"走到最近的椅子旁边"}`，
或离线测试用的 `{"target":"真实对象ID"}`；`POST /api/stop` 接收 `{}`。

对运行中的服务做两条真实导航任务的连续性验证（会移动机器人）：

```bash
/home/caosiqi/miniconda3/envs/habitat/bin/python -m scripts.checks.check_live
# 同时验证停止当前任务、清空排队任务：
/home/caosiqi/miniconda3/envs/habitat/bin/python -m scripts.checks.check_live --check-stop
```

本机已通过 HTTP、JPEG、MJPEG 及两条顺序任务的实际仿真验证，第二条起点与第一条终点完全一致。
停止与队列清空也已通过实测；当前共 10 项单元测试通过。
自然语言会话历史、澄清续答和模型失败恢复另有 mock 单元测试；真实模型服务仍需配置后验证。

### 单次命令行运行

从 `/home/caosiqi/project/habitat_work` 执行，使用已安装 Habitat-Sim 0.3.3 的环境：

```bash
# 只导出场景对象、目标站立点、路标及连接。
/home/caosiqi/miniconda3/envs/habitat/bin/python -m legacy_gt.run --mode export

# 离线执行验证：BFS 选路线，人形逐段执行。不调用模型、不声称是 agent 规划。
/home/caosiqi/miniconda3/envs/habitat/bin/python -m legacy_gt.run --mode smoke

# 验证规划输出边界，无需 GPU 或模型服务。
/home/caosiqi/miniconda3/envs/habitat/bin/python -m unittest discover -s tests -t . -p 'test_*.py'
```

渲染需要 GPU/EGL 可访问；受限沙箱中可能提示 `no EGL devices found`。
输出放在 `outputs/<时间>/`，每次独立目录，包含 `scene.json`、`plan_0.json`、
`walk.mp4`、`final.png`、`trajectory.json` 和 `result.json`。
`result.json` 是执行成功与否的依据，不仅看视频是否生成。

本机已验证：

- `outputs/20261013_174802_451479/`：5 段路线到 `chair_04_:0000`，终点误差约 0.0775 米。
- `outputs/20261013_174858_454683/`：6 段含转弯路线到 `chair_05_:0000`，终点误差约 0.0598 米。
- 6 项规划边界及模型 HTTP 请求契约测试通过；HTTP 测试使用 mock，未调用真实模型。
- 模型服务尚未配置，因此自然语言规划效果与真实模型重规划尚未实测。

现有数据集配置还引用了一些本地缺失的 `v3_sc4_staging_*` NavMesh，
加载时会打印相关错误；本 demo 的 `apt_0` 已成功加载并重新计算 NavMesh。

可以通过 `--target` 指定导出对象的完整 ID，通过 `--start` 指定路标 ID。
`--mode replay --plan <plan.json>` 执行已有路线，并重新验证当前场景中的连接。
路线文件需来自同一场景和地图参数；不要混用不同场景的节点 ID。

## 真正接入 agent

实现的是可替换的 chat-completions HTTP 客户端，模型自己输出节点路线。
配置提供商对应的 API 根地址（通常包含 `/v1`）和模型名称：

```bash
export NAV_LLM_BASE_URL='http://127.0.0.1:8000/v1'
export NAV_LLM_MODEL='你的模型名称'
# 需要认证时，在本地环境变量设置 NAV_LLM_API_KEY，勿提交密钥。
/home/caosiqi/miniconda3/envs/habitat/bin/python -m legacy_gt.run \
  --mode agent --instruction '走到最近的一把椅子旁边'
```

本地服务示例地址不是自动部署的模型。实际模型效果需要接上服务后验证。
不提供图片，模型接收指令、GT 对象表和路标图。模型调用不使用 Habitat 全局路径查询。
输出必须是 `{"target_object":"真实对象ID","route":["起点","中间节点","目标站立点"]}`。
默认最多重规划两次；无效路线反馈给模型，不直接执行。遇到阻塞时以实际位置生成新起点。
目标属性未知或目标有歧义时，模型可返回 `needs_clarification`；程序保存问题并停止，
当前 CLI 尚未实现交互追问续跑。网络错误会明确报错，不自动切换成假模型或 BFS。
浏览器版 `serve.py` 已支持持续会话和接着回答澄清问题。

颜色不从 semantic_id 推断。`legacy_gt/annotations.json` 初始为空；看过资产后可人工补充：

```json
{
  "从 scene.json 复制的真实对象ID": {"color": "red"}
}
```

这只是字段示例，并未声明场景中任何椅子是红色。补充属性前，
“红色椅子”无法由现有 GT 确定；建议先使用类别、实例 ID 或最近目标指令。

## 模块与分工

- `world.py`：读取加载后的刚体状态及语义词典，变换包围盒到世界坐标，按身体半径重建 NavMesh，生成直连路标图。
- `planning.py`：模型规划接口、路线结构和连接校验，以及明确区分的离线 BFS 测试基线。
- `executor.py`：复用 Habitat-Lab 官方行走控制器，逐帧校验根部移动并跟踪指定路段，不计算绕路。
- `legacy_gt/run.py`：有限次数的规划、执行、反馈循环，保存视频和日志。
- `session.py`：持续会话、历史上下文、从实际位置连接路标图、单次失败隔离。
- `task_graph.py`：LangGraph 多步骤任务、步骤校验、保存起点、返回与顺序执行。
- `serve.py` / `legacy_gt/web.html`：本地 HTTP 服务、MJPEG 视频流、浏览器指令队列与停止控制。所有仿真操作只在同一个工作线程运行。

以后换 nanobot/Harness：调用这些普通 Python 接口或加 HTTP/MCP 包装即可。
以后接视觉：替换对象信息来源；若连全局 GT 地图也移除，还需接建图与机器人定位。

## 当前边界

- 家具和杂物冻结，门保持加载时状态；不支持动态场景地图自动更新。
- 导出的目标对象当前为刚体家具；关节家具参与几何导航，但没有作为语义任务目标导出。
- 路标图使用规则采样和家具周围候选点，只保留最大连通分量；它不保证覆盖整个 NavMesh。
- “旁边”定义为站到关联可达节点，候选点按水平包围盒边界距离筛选；没有实现“左边”等相对空间关系。
- 不执行最终面向物体动作；当前成功指标是到达目标站立点 0.1 米内。
- NavMesh 使用圆柱体近似并逐步过滤根部移动；这不保证手臂、步态摆动的完整碰撞包络。
- 语言目标匹配由模型承担，程序能验证路线与对象 ID 的一致性，不能独立保证语言语义理解正确。

## 依赖与资产来源

复用 `../habitat-lab/habitat-lab`，缺失依赖安装于 `vendor/`：
`hydra-core==1.3.2`、`omegaconf==2.3.0`、`antlr4-python3-runtime==4.9.3`、
`gym==0.23.1`、`gym-notices==0.1.0`。不覆盖 conda 环境已有包。
还使用已有的 numpy、magnum、imageio、imageio-ffmpeg 等 Habitat 环境依赖。

人形资产取自官方 <https://huggingface.co/datasets/ai-habitat/habitat_humanoids> 的 `female_0/`：
URDF、GLB、AO 配置和动作 pickle。使用这些资产须遵循上游资产许可。
只加载可信的官方 pickle；已下载的三个 LFS 文件 SHA256 与上游记录匹配：

```text
female_0.urdf: beefa71417e13eadf0e93a9689581d7ffec37586d062bb8118fda6f840f8c536
female_0.glb: 7e007a0dcb79c986a4f9570234b649563669e8c200e4425f5cf70a025c12c28a
female_0_motion_data_smplx.pkl: 601b8b210e00c73bf26e6ec9d5e2573d2e8b1f815a2e2e172fc1ae53262c69d8
```
