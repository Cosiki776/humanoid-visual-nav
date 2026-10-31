#!/usr/bin/env python3
"""Matched end-to-end model timing; preserves raw logs and request traces."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--instructions', nargs='+', default=['找柜子', '找沙发'])
    p.add_argument('--timeout', type=float, default=1800)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    models = [('cloud', 'qwen3.8-max'), ('local27b', 'qwen3.8:27b-q4_K_M'), ('local9b', 'qwen3.5:9b')]
    starts = {}
    rows = []
    for label, model in models:
        for task, instruction in enumerate(args.instructions, 1):
            name = f'{label}_task{task}'
            dest = out / name
            env = os.environ.copy()
            for key in ('NAV_VLM_BASE_URL', 'NAV_VLM_API_KEY'):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES='0', NAV_VLM_BACKEND='openai' if label == 'cloud' else 'ollama',
                       NAV_VLM_TIMEOUT='300', NAV_VLM_MAX_TOKENS='1024', NAV_VLM_NUM_CTX='16384',
                       NO_PROXY='dashscope.aliyuncs.com,127.0.0.1,localhost',
                       no_proxy='dashscope.aliyuncs.com,127.0.0.1,localhost',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            if label != 'cloud':
                env.update(NAV_VLM_BASE_URL='http://127.0.0.1:11434', NAV_VLM_API_KEY='')
            cmd = [sys.executable, 'run_visual.py', '--realtime', '--execution-mode', 'async',
                   '--scene', 'apt_0', '--seed', '7', '--instruction', instruction,
                   '--vlm-model', model, '--max-observations', '24', '--output', str(dest)]
            if task in starts:
                cmd += ['--start', *map(str, starts[task])]
            row = dict(model=model, task=task, instruction=instruction, output=str(dest))
            print('START '+name, flush=True)
            begin = time.perf_counter()
            with (out / (name+'.log')).open('w') as log:
                try:
                    row['exit_code'] = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                                     stderr=subprocess.STDOUT, timeout=args.timeout).returncode
                except subprocess.TimeoutExpired:
                    row['error'] = 'process timeout'
            row['process_seconds'] = time.perf_counter()-begin
            if (dest/'result.json').exists():
                result = json.loads((dest/'result.json').read_text())
                for key in ('status', 'verified', 'elapsed_seconds', 'observations', 'path_length_m', 'start_position', 'vlm_calls'):
                    row[key] = result.get(key)
                if result.get('start_position') is not None:
                    starts.setdefault(task, result['start_position'])
                row['same_start'] = result.get('start_position') == starts.get(task)
            timings = dest/'vlm_requests'/'timings.jsonl'
            metrics = [json.loads(line) for line in timings.read_text().splitlines()] if timings.exists() else []
            row['model_seconds'] = sum(m['wall_seconds'] for m in metrics)
            row['load_seconds'] = sum(m.get('load_duration_seconds', 0) for m in metrics) if label != 'cloud' else None
            row['model_requests'] = len(metrics)
            row['failed_requests'] = sum(not m['ok'] for m in metrics)
            row['first_request_seconds'] = metrics[0]['wall_seconds'] if metrics else None
            rows.append(row)
            (out/'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            print(json.dumps(row, ensure_ascii=False), flush=True)
    lines = ['# 本地与云端导航耗时', '', '每个模型每条任务实跑一次；apt_0、seed=7、异步导航、实时行走、24 次观察上限。',
             '总进程时间包含初始化和退出；模型累计时间可能与异步行走重叠。加载时间仅本地接口提供，首次请求不保证冷启动。',
             '状态为系统自报，不代表独立真值评测；单次测试不能代表稳定速度或成功率。', '',
             '| 模型 | 任务 | 状态 | 总耗时(s) | 导航循环(s) | 模型累计(s) | 本地加载(s) | 请求数 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['instruction']} | {r.get('status', r.get('error', 'error'))} | {r['process_seconds']:.2f} | {r.get('elapsed_seconds', 0):.2f} | {r['model_seconds']:.2f} | {r['load_seconds'] if r['load_seconds'] is not None else 'N/A'} | {r['model_requests']} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
