#!/usr/bin/env python3
"""Sequential complete navigation trials, separate simulator GPU and Ollama service."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instruction', default='走到白色椅子旁边')
    parser.add_argument('--scene', default='apt_0')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--max-observations', type=int, default=24)
    parser.add_argument('--sim-gpu', default='0')
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--execution-mode', choices=['sync', 'async'], default='async')
    parser.add_argument('--warm-local', action='store_true', help='Preload the local model before its trial')
    parser.add_argument('--include-process-timing', action='store_true',
                        help='Include startup/teardown timing fields in summary.json; full data is always in diagnostics.json')
    args = parser.parse_args()
    if args.sim_gpu == '2':
        parser.error('GPU 2 is reserved for Ollama; choose another simulator GPU')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(instruction=args.instruction, scene=args.scene, seed=args.seed,
                    simulator_gpu=args.sim_gpu, execution_mode=args.execution_mode,
                    realtime=True, web_video=False)
    details = dict(**metadata, trials=[])
    summary = dict(**metadata,
                   timing_scope='full_process_diagnostics' if args.include_process_timing else 'navigation_loop_and_llm_requests',
                   trials=[])
    common = os.environ.copy()
    common.update(CUDA_VISIBLE_DEVICES=args.sim_gpu,
                  NO_PROXY='dashscope.aliyuncs.com,127.0.0.1,localhost',
                  no_proxy='dashscope.aliyuncs.com,127.0.0.1,localhost',
                  NAV_VLM_TIMEOUT='300', NAV_VLM_MAX_TOKENS='1024')
    for backend in ['cloud', 'local']:
        env = common.copy()
        env.pop('NAV_VLM_BASE_URL', None)
        env.pop('NAV_VLM_API_KEY', None)
        env['NAV_VLM_BACKEND'] = 'openai' if backend == 'cloud' else 'ollama'
        model = 'qwen3.8-max' if backend == 'cloud' else 'qwen3.8:27b-q4_K_M'
        if backend == 'local':
            env.update(NAV_VLM_BASE_URL='http://127.0.0.1:11434', NAV_VLM_API_KEY='', NAV_VLM_NUM_CTX='16384')
            if args.warm_local:
                from scripts.utilities.warmup_vlm import preload
                warmup = preload(env['NAV_VLM_BASE_URL'], model, int(env['NAV_VLM_NUM_CTX']), args.timeout)
            else:
                warmup = None
        output = args.output/backend
        command = [sys.executable, 'run_visual.py', '--execution-mode', args.execution_mode,
                   '--realtime', '--scene', args.scene,
                   '--seed', str(args.seed), '--instruction', args.instruction,
                   '--vlm-model', model, '--device', 'cuda:0',
                   '--max-observations', str(args.max_observations), '--output', str(output)]
        # Fix the second run to the first run's actual snapped start, not just its RNG seed.
        if details['trials'] and details['trials'][0].get('start_position'):
            command += ['--start']+list(map(str, details['trials'][0]['start_position']))
        row = dict(backend=backend, model=model, command=command)
        if backend == 'local':
            row['warmup'] = warmup
        print('Starting '+backend, flush=True)
        begin = time.perf_counter()
        with (args.output/(backend+'.log')).open('w') as log:
            try:
                completed = subprocess.run(command, env=env, cwd=Path(__file__).resolve().parents[2],
                                           stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                row['exit_code'] = completed.returncode
            except subprocess.TimeoutExpired:
                row['error'] = 'process timeout'
        row['process_wall_seconds'] = time.perf_counter()-begin
        if (output/'result.json').exists():
            result = json.loads((output/'result.json').read_text())
            row.update({key:result.get(key) for key in ['status','verified','selected','start_position',
                        'final_position','observations','path_length_m','elapsed_seconds','vlm_calls']})
            timings = result.get('vlm_timings', [])
            row['llm_wall_seconds'] = sum(t['wall_seconds'] for t in timings)
            row['llm_load_seconds'] = sum(t.get('load_duration_seconds',0) for t in timings)
            row['llm_instruction_seconds'] = sum(t['wall_seconds'] for t in timings if t['phase']=='instruction')
            row['llm_decision_seconds'] = sum(t['wall_seconds'] for t in timings if t['phase']=='decision')
            row['non_llm_process_seconds'] = row['process_wall_seconds']-row['llm_wall_seconds']
            for stage in ['capture','perception','memory']:
                row[stage+'_seconds'] = sum(e.get(stage+'_seconds',0) for e in result['events'] if e['type']=='observation')
        if (output/'models.json').exists():
            setup = json.loads((output/'models.json').read_text())
            row.update(setup_seconds=setup['setup_seconds'], instruction_parse_seconds=setup['instruction_parse_seconds'])
        if (output/'config.json').exists():
            config = json.loads((output/'config.json').read_text())
            row['execution_mode'] = config.get('execution_mode')
        details['trials'].append(row)
        navigation_row = {key:row.get(key) for key in
                          ('backend', 'model', 'status', 'verified', 'selected', 'observations',
                           'path_length_m', 'vlm_calls', 'start_position', 'final_position')}
        navigation_row['execution_mode'] = row.get('execution_mode')
        navigation_row['navigation_seconds'] = row.get('elapsed_seconds')
        navigation_row['llm_request_seconds'] = row.get('llm_wall_seconds')
        navigation_row['llm_decision_seconds'] = row.get('llm_decision_seconds')
        navigation_row['llm_load_seconds'] = row.get('llm_load_seconds')
        summary['trials'].append(row if args.include_process_timing else navigation_row)
        (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        (args.output/'diagnostics.json').write_text(json.dumps(details, ensure_ascii=False, indent=2))
        print(json.dumps(navigation_row, ensure_ascii=False), flush=True)
    same_start = (len(details['trials']) == 2 and
        details['trials'][0].get('start_position') is not None and
        details['trials'][0]['start_position'] == details['trials'][1].get('start_position'))
    summary['same_start'] = details['same_start'] = same_start
    summary['limitations'] = ('Navigation timing is the complete loop wall time and includes any overlap between movement and LLM requests. '
                              'LLM request time is reported separately and is not subtracted in async mode. '
                              'One run per model; status/verified are system claims, not independent GT success.')
    details['limitations'] = ('Diagnostic timings include startup, teardown, model loading and caching. '
                              'Sequential runs may have different loading/cache costs and decisions.')
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.output/'diagnostics.json').write_text(json.dumps(details, ensure_ascii=False, indent=2))
    lines = ['# 异步导航耗时对比', '',
             '导航秒数是完整导航循环墙钟时间；LLM 请求累计时间单独报告。异步模式下二者可能重叠，不做相减。', '',
             '| 后端 | 模型 | 状态 | 完整导航循环秒数 | LLM请求秒数 | LLM加载秒数 | 观察数 | 模型调用数 | 路径米数 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['trials']:
        navigation = row.get('navigation_seconds', row.get('elapsed_seconds'))
        requests = row.get('llm_request_seconds', row.get('llm_wall_seconds'))
        load = row.get('llm_load_seconds', 0)
        lines.append('| {backend} | {model} | {status} | {navigation:.2f} | {requests:.2f} | {load:.2f} | {observations} | {vlm_calls} | {path_length_m:.2f} |'.format(
            navigation=navigation, requests=requests, load=load, **row))
    lines.extend(['', '完整启动、模型加载和阶段耗时保存在 `diagnostics.json`。'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
