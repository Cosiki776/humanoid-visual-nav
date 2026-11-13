# 一次真实 VLM 调用：去另一个白色椅子旁边

来源运行：`web_visual_20261106_134206_398685`；调用编号：`0007`。

这是历史调用的直接导出，没有重新请求模型，也没有用当前代码重建输入。

用户原任务是“去另一个白色椅子旁边”。本次视觉核实实际收到的任务字段见下方 JSON；原指令不一定逐字出现在该次请求中。

## 1. 系统提示词（原文）

```text
核实黄色框候选，同时留意全图中更符合任务的物体。青色框为其他候选，框上ID与visible_candidates对应。类别标签不是事实。判断目标类别、主体颜色/属性是否符合；遮挡不等于类别错误，可以结合历史清晰图理解物体，但不能用历史图替代当前到达确认。光照、遮挡或画面不清时回答unknown，不猜测。无属性要求时attributes_match=yes。返回JSON：{"candidate_id":"指定ID","observation_id":"当前观测ID","category_match":"yes|no|unknown","attributes_match":"yes|no|unknown","reason":"可见依据", "approach_recommended":false, "target_hint":null}。如果能明确辨认类别或目标属性，值得靠近核实另一项，approach_recommended可为true，并说明具体依据；仅有模糊暗块、类别和属性都unknown时不要建议直接靠近；明确不符时不能建议靠近。如果别的框更明确符合目标，target_hint返回{"candidate_id":"该框ID","category_match":"yes","attributes_match":"yes","reason":"具体位置和可见依据"}。不要只在reason里提到其他目标，要填写target_hint。只推荐当前可见且未excluded的ID，不得编造ID。
```

## 2. 用户消息（严格按原始内容顺序）

### 内容块 1

```json
{
  "observation_id": "obs_00014",
  "candidate_id": "v0018",
  "detected_category": "chair",
  "target_categories": [
    "chair"
  ],
  "attributes": [
    "white"
  ],
  "identity_state": "tentative",
  "visible_candidates": [
    {
      "id": "v0017",
      "detected_category": "chair",
      "bbox": [
        159.72064208984375,
        302.0284423828125,
        230.26541137695312,
        457.97760009765625
      ],
      "excluded": false
    },
    {
      "id": "v0018",
      "detected_category": "chair",
      "bbox": [
        705.606201171875,
        252.0787353515625,
        760.3946533203125,
        290.5179443359375
      ],
      "excluded": false
    }
  ],
  "excluded_previous_id": "v0001"
}
```

### 内容块 2

```text
Current scene. Yellow box identifies candidate v0018; detected category: chair (detector hypothesis, verify visually).
```

### 内容块 3

![实际发送图片 1](images/01.jpg)

### 内容块 4

```text
Current candidate v0018 from obs_00014
```

### 内容块 5

![实际发送图片 2](images/02.jpg)

### 内容块 6

```text
Historical previous target v0001. Similar appearance alone does not establish identity.
```

### 内容块 7

![实际发送图片 3](images/03.jpg)

## 3. 模型输出（原文）

```json
{
  "candidate_id": "v0018",
  "observation_id": "obs_00014",
  "category_match": "unknown",
  "attributes_match": "unknown",
  "reason": "黄色框v0018位于画面右侧边缘，图像非常模糊且被截断，仅能看到深色和浅色色块，无法辨认是否为椅子，也无法确认颜色属性。",
  "approach_recommended": false,
  "target_hint": {
    "candidate_id": "v0017",
    "category_match": "yes",
    "attributes_match": "yes",
    "reason": "青色框v0017位于冰箱左侧，清晰显示为一把白色椅子，类别和颜色属性均符合目标要求。"
  }
}
```

## 4. 调用统计

```json
[
  {
    "call": 7,
    "backend": "openai",
    "model": "qwen3.8-max",
    "image_count": 3,
    "phase": "decision",
    "ok": true,
    "usage": {
      "completion_tokens": 180,
      "completion_tokens_details": {
        "text_tokens": 180
      },
      "prompt_tokens": 1149,
      "prompt_tokens_details": {
        "cached_tokens": 0,
        "image_tokens": 540,
        "text_tokens": 609
      },
      "total_tokens": 1329
    },
    "finish_reason": "stop",
    "wall_seconds": 4.220157256349921
  }
]
```

## 文件说明与边界

- `request_original.json`：原始日志，含实际发送图片的 Base64。
- `request_readable.json`：同样的消息顺序，图片 Base64 替换为本地文件路径，仅供阅读。
- `response_original.json`：服务端完整返回；`model_output.txt` 是模型回答原文。
- `images/`：直接解码请求内图片，没有重新绘图或压缩。
- `manifest.json`：来源和 SHA-256 校验值。
- 请求日志保存 system/content，不是完整 HTTP 报文；不包含密钥或请求头。
- 这是视觉判断调用，不是到达成功证明；模型判断与程序后续移动是两个环节。
