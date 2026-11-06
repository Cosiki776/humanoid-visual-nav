"""Export one saved multimodal call without re-running inference."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil


def export(run, call, output):
    trace = run / 'vlm_requests'
    request_path = trace / f'request_{call:04d}.json'
    response_path = trace / f'response_{call:04d}.json'
    request = json.loads(request_path.read_text())
    response = json.loads(response_path.read_text())
    result_path = run / 'result.json'
    instruction = json.loads(result_path.read_text()).get('instruction', '见输入任务字段') if result_path.exists() else '见输入任务字段'
    output.mkdir(parents=True, exist_ok=False)
    images = output / 'images'
    images.mkdir()
    manifest = dict(source_run=str(run.resolve()), call=call, files={})
    for source, name in [(request_path, 'request_original.json'),
                         (response_path, 'response_original.json')]:
        shutil.copyfile(source, output / name)
        manifest['files'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / 'system_prompt.txt').write_text(request['system'], encoding='utf-8')
    raw = (response['choices'][0]['message']['content'] if 'choices' in response
           else response['message']['content'])
    (output / 'model_output.txt').write_text(raw, encoding='utf-8')
    timings = [json.loads(line) for line in (trace / 'timings.jsonl').read_text().splitlines()
               if line.strip() and json.loads(line)['call'] == call]
    (output / 'timing.json').write_text(json.dumps(timings, ensure_ascii=False, indent=2))
    lines = [f'# 一次真实 VLM 调用：{instruction}', '',
             f'来源运行：`{run.name}`；调用编号：`{call:04d}`。', '',
             '这是历史调用的直接导出，没有重新请求模型，也没有用当前代码重建输入。', '',
             f'用户原任务是“{instruction}”。本次视觉核实实际收到的任务字段见下方 JSON；'
             '原指令不一定逐字出现在该次请求中。', '',
             '## 1. 系统提示词（原文）', '', '```text', request['system'], '```', '',
             '## 2. 用户消息（严格按原始内容顺序）', '']
    readable = []
    image_index = 0
    content = request['content']
    if isinstance(content, str):
        content = [dict(type='text', text=content)]
    for index, part in enumerate(content, 1):
        lines += [f'### 内容块 {index}', '']
        if part['type'] == 'text':
            text = part['text']
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                lines += ['```text', text, '```', '']
            else:
                lines += ['```json', json.dumps(parsed, ensure_ascii=False, indent=2), '```', '']
                if index == 1:
                    (output / 'input_context.json').write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
            readable.append(part)
        elif part['type'] == 'image_url':
            image_index += 1
            header, encoded = part['image_url']['url'].split(',', 1)
            if not header.startswith('data:image/') or ';base64' not in header:
                raise ValueError('Expected recorded inline image')
            extension = 'png' if header.startswith('data:image/png') else 'jpg'
            name = f'images/{image_index:02d}.{extension}'
            payload = base64.b64decode(encoded, validate=True)
            (output / name).write_bytes(payload)
            manifest['files'][name] = hashlib.sha256(payload).hexdigest()
            readable.append(dict(type='image_file', path=name))
            lines += [f'![实际发送图片 {image_index}]({name})', '']
        else:
            raise ValueError('Unsupported content block')
    (output / 'request_readable.json').write_text(json.dumps(
        dict(system=request['system'], content=readable), ensure_ascii=False, indent=2))
    lines += ['## 3. 模型输出（原文）', '', '```json', raw, '```', '',
              '## 4. 调用统计', '', '```json', json.dumps(timings, ensure_ascii=False, indent=2), '```', '',
              '## 文件说明与边界', '',
              '- `request_original.json`：原始日志，含实际发送图片的 Base64。',
              '- `request_readable.json`：同样的消息顺序，图片 Base64 替换为本地文件路径，仅供阅读。',
              '- `response_original.json`：服务端完整返回；`model_output.txt` 是模型回答原文。',
              '- `images/`：直接解码请求内图片，没有重新绘图或压缩。',
              '- `manifest.json`：来源和 SHA-256 校验值。',
              '- 请求日志保存 system/content，不是完整 HTTP 报文；不包含密钥或请求头。',
              '- 这是视觉判断调用，不是到达成功证明；模型判断与程序后续移动是两个环节。', '']
    manifest['image_count'] = image_index
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(output=str(output.resolve()), image_count=image_index,
                          model_output=raw), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--call', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.call, args.output)
