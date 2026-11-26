# Humanoid Visual Navigation

基于 Habitat 的人形视觉导航实验系统。系统在 ReplicaCAD 场景中运行运动学人形角色，使用 RGB-D 观察、开放词汇检测/分割、视图记忆、主动搜索和到达核验完成自然语言导航，并提供浏览器工作台。

> 本仓库不是开箱即用的独立仿真包。代码可以从 GitHub 获取，但 **Habitat-Sim、Habitat-Lab、ReplicaCAD、人形资产和视觉模型权重需要另外安装或下载**。它们体积较大或受各自许可证约束，因此没有提交到本仓库。

## 功能与边界

- 浏览器中的连续视觉导航会话和实时画面
- 本地 Ollama VLM 或 OpenAI-compatible 云端 VLM
- RGB-D 观察、目标实例记忆、主动补视角和几何到达核验
- Habitat `KinematicHumanoid` 路径执行
- GT 基线、离线测试和评测脚本

当前系统是仿真研究原型，不是动力学平衡控制器，也没有真实机器人部署接口。默认场景为 ReplicaCAD `apt_0`；当前主流程并不使用 HM3D，所以复现默认演示 **不需要下载 HM3D**。

## 已验证环境

以下组合是本项目当前实际使用并验证过的基线：

| 组件 | 版本 |
| --- | --- |
| OS | Linux x86_64 |
| Python | 3.9.19 |
| Habitat-Sim | 0.3.3，启用 Bullet；服务器建议 headless/EGL |
| Habitat-Lab | `stable`，本机提交 `0fb6f43ff` |
| NumPy | 1.26.4 |
| PyTorch / Torchvision | 2.0.1 / 0.15.2 |
| CUDA（本机 PyTorch 构建） | 11.7 |

其他版本可能可用，但未经过本项目完整验证。渲染模式需要 Linux GPU 和可用的 EGL；仅运行不导入 Habitat 的单元测试不需要 GPU。

## 1. 克隆代码并创建环境

建议让项目、Habitat-Lab 和数据目录互相独立，不要把 Conda 环境或场景数据复制进本仓库。

```bash
git clone https://github.com/Cosiki776/humanoid-visual-nav.git
cd humanoid-visual-nav

conda create -n humanoid-nav python=3.9 -y
conda activate humanoid-nav
```

安装带 Bullet 的 Habitat-Sim。远程服务器通常使用 headless/EGL 版本：

```bash
conda install habitat-sim=0.3.3 withbullet headless \
  -c conda-forge -c aihabitat -y
```

有桌面显示环境时，可以去掉 `headless`。安装后检查：

```bash
python -c "import habitat_sim; print(habitat_sim.__version__)"
```

Habitat-Sim 官方安装说明：<https://github.com/facebookresearch/habitat-sim#installation>。

## 2. 安装 Habitat-Lab

人形控制器来自 Habitat-Lab，因此只安装 Habitat-Sim 不够。推荐克隆 `stable` 分支并以 editable 模式安装：

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git ../habitat-lab
python -m pip install -e ../habitat-lab/habitat-lab
```

如果 Habitat-Lab 位于其他位置，启动前指定：

```bash
export HABITAT_LAB_PATH=/absolute/path/to/habitat-lab/habitat-lab
```

## 3. 安装项目依赖

视觉模式使用 PyTorch、Transformers、Segment Anything 等依赖：

```bash
python -m pip install -r requirements/visual.txt
python -m pip install -r requirements/graph.txt
```

`requirements/visual.txt` 记录的是已验证版本。若机器已有与 CUDA 匹配的 PyTorch，请保留该 PyTorch，并单独安装其余包，避免 `pip` 覆盖可工作的 CUDA 构建。

## 4. 下载 ReplicaCAD 场景

默认演示使用 ReplicaCAD，而不是 HM3D。按照 Habitat 官方数据说明下载 ReplicaCAD：

```bash
python -m habitat_sim.utils.datasets_download \
  --uids replica_cad_dataset \
  --data-path ../data
```

项目默认期望下面的文件存在：

```text
../data/replica_cad/replicaCAD.scene_dataset_config.json
```

如果数据放在别处，设置数据根目录（该目录下应有 `replica_cad/`）：

```bash
export HABITAT_DATA_PATH=/absolute/path/to/data
```

ReplicaCAD 的下载、目录结构和许可证以 Habitat-Lab 官方说明为准：<https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md>。

## 5. 下载人形资产

运行导航必须具备 `female_0` 的 URDF、GLB、动作数据和配置文件。资产来自官方数据集：
<https://huggingface.co/datasets/ai-habitat/habitat_humanoids>。

将以下文件放入 `assets/female_0/`：

```text
assets/female_0/
├── female_0.ao_config.json
├── female_0.glb
├── female_0.urdf
└── female_0_motion_data_smplx.pkl
```

也可以把资产放在仓库外，并设置：

```bash
export NAV_HUMANOID_ASSETS=/absolute/path/to/female_0
```

这些资产不会被 Git 上传；使用时请遵循上游许可证。动作文件是 pickle，只应从可信的官方来源加载。

## 6. 下载视觉权重

完整视觉流程默认需要 Segment Anything ViT-B checkpoint：

```text
weights/
└── sam_vit_b_01ec64.pth
```

- `sam_vit_b_01ec64.pth`：Segment Anything ViT-B checkpoint

Grounding DINO 默认模型为 `IDEA-Research/grounding-dino-tiny`，由 Transformers/Hugging Face 在首次使用时下载并缓存；离线服务器应提前准备该模型缓存，或通过 `--dino-model` 指向本地模型目录。

项目默认从相邻目录 `../weights/` 读取。权重在其他位置时设置：

```bash
export NAV_WEIGHTS_PATH=/absolute/path/to/weights
```

模型权重没有提交到本仓库。只运行纯逻辑单元测试或部分 GT 基线时不需要这些权重。

## 7. 配置视觉语言模型

### 方案 A：云端 OpenAI-compatible API

复制示例配置：

```bash
cp config/model_config.example.json config/model_config.json
```

编辑 `config/model_config.json`：

```json
{
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model": "qwen3.8-max",
  "api_key": "YOUR_API_KEY",
  "timeout_seconds": 120
}
```

`config/model_config.json` 已被 `.gitignore` 排除。不要提交 API Key。也可以使用环境变量，且环境变量优先：

```bash
export NAV_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export NAV_LLM_MODEL=qwen3.8-max
export NAV_LLM_API_KEY=YOUR_API_KEY
```

### 方案 B：本地 Ollama

先在本机安装并启动 Ollama，再准备 `start_visual.sh` 中配置的模型。当前默认模型名是：

```text
qwen3.8:27b-q4_K_M
```

该模型需要较大的显存/内存；资源不足时可通过 `NAV_VLM_MODEL` 选择兼容的视觉模型。

## 8. 启动与验证

先检查关键依赖：

```bash
python -c "import habitat_sim, habitat; print('Habitat imports OK')"
test -f "${HABITAT_DATA_PATH:-../data}/replica_cad/replicaCAD.scene_dataset_config.json"
test -f "${NAV_HUMANOID_ASSETS:-assets/female_0}/female_0.urdf"
```

启动云端 VLM 模式：

```bash
bash start_visual.sh cloud
```

启动本地 Ollama 模式：

```bash
bash start_visual.sh local
```

也可直接启动服务：

```bash
python serve.py --mode visual --scene apt_0 --port 8770
```

浏览器打开 <http://127.0.0.1:8770>。远程服务器需要用 VS Code Ports 或 SSH 转发端口


服务默认只绑定 `127.0.0.1`。输出写入 `outputs/`，该目录不会上传 GitHub。

## 9. 测试

运行不依赖真实模型服务的单元测试：

```bash
python -m unittest discover -s tests -t . -p 'test_*.py'
```

运行单次视觉导航（需要 Habitat、场景、人形资产、视觉权重和 VLM）：

```bash
python run_visual.py --scene apt_0 --instruction "走到一把白色椅子旁边"
```

查看其他参数：

```bash
python run_visual.py --help
python serve.py --help
```

## 外部资源清单

| 资源 | 是否必需 | 是否包含在仓库 | 用途 |
| --- | --- | --- | --- |
| Habitat-Sim 0.3.3 + Bullet | 必需 | 否 | 仿真、物理、渲染和 NavMesh |
| Habitat-Lab stable | 必需 | 否 | 人形角色和控制器 |
| ReplicaCAD | 默认演示必需 | 否 | `apt_0` / `apt_1` 场景 |
| HM3D | 默认演示不需要 | 否 | 仅扩展到 HM3D 场景时需要 |
| `female_0` humanoid assets | 必需 | 否 | 人形模型和动作 |
| SAM checkpoint / Grounding DINO | 完整视觉模式必需 | 否 | 分割和开放词汇检测 |
| 云端 VLM Key 或本地 Ollama | 自然语言视觉模式必需 | 否 | 目标理解和视觉决策 |

## 目录结构

```text
app/                  浏览器后端与配置读取
config/               示例配置（本地密钥配置被忽略）
legacy_gt/            GT 基线与历史流程
requirements/         Python 依赖版本
scripts/              检查、评测和实用脚本
simulation/           Habitat 世界与人形执行器
tests/                单元测试
visual_nav/           视觉导航、搜索、记忆和核验
web/                  浏览器页面
run_visual.py         单次视觉导航入口
run_visual_session.py 连续 CLI 会话入口
serve.py              Web 服务入口
start_visual.sh       本地/云端 VLM 启动脚本
```

## 常见问题

### `ModuleNotFoundError: habitat_sim`

当前 Conda 环境没有安装 Habitat-Sim，或运行命令时没有激活正确环境。

### `ModuleNotFoundError: habitat`

缺少 Habitat-Lab。执行 editable 安装，或设置 `HABITAT_LAB_PATH`。

### 找不到 `replicaCAD.scene_dataset_config.json`

ReplicaCAD 未下载到默认目录。检查 `HABITAT_DATA_PATH` 是否指向包含 `replica_cad/` 的数据根目录。

### `no EGL devices found`

服务器没有向当前进程暴露可用 GPU/EGL，或安装的 Habitat-Sim 不是合适的 headless 构建。先检查 `nvidia-smi` 和容器/任务的 GPU 权限。

### 找不到 SAM checkpoint 或 humanoid asset

这些大文件不会从 GitHub 自动获得。按照上面的资源步骤下载，并检查 `NAV_WEIGHTS_PATH`、`NAV_HUMANOID_ASSETS`。

## 安全与数据说明

- 不要提交 `config/model_config.json`、`.env`、API Key 或服务凭据。
- `outputs/` 可能包含输入指令、模型响应、观察图像和轨迹，默认不进入 Git。
- ReplicaCAD、人形资产和模型权重受各自上游许可证约束，本仓库不重新分发这些资源。
