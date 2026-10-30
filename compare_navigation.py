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
    args = parser.parse_args()
    if args.sim_gpu == '2':
        parser.error('GPU 2 is reserved for Ollama; choose another simulator GPU')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(instruction=args.instruction, scene=args.scene, seed=args.seed,
                   simulator_gpu=args.sim_gpu, realtime=True, web_video=False, trials=[])
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
        output = args.output/backend
        command = [sys.executable, 'run_visual.py', '--realtime', '--scene', args.scene,
                   '--seed', str(args.seed), '--instruction', args.instruction,
                   '--vlm-model', model, '--device', 'cuda:0',
                   '--max-observations', str(args.max_observations), '--output', str(output)]
        # Fix the second run to the first run's actual snapped start, not just its RNG seed.
        if summary['trials'] and summary['trials'][0].get('start_position'):
            command += ['--start']+list(map(str, summary['trials'][0]['start_position']))
        row = dict(backend=backend, model=model, command=command)
        print('Starting '+backend, flush=True)
        begin = time.perf_counter()
        with (args.output/(backend+'.log')).open('w') as log:
            try:
                completed = subprocess.run(command, env=env, cwd=Path(__file__).resolve().parent,
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
        summary['trials'].append(row)
        (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary['same_start'] = (len(summary['trials']) == 2 and
        summary['trials'][0].get('start_position') is not None and
        summary['trials'][0]['start_position'] == summary['trials'][1].get('start_position'))
    summary['limitations'] = 'One independent run per model. Status/verified are system claims, not independent GT success. Sequential runs may have different loading/cache costs and decisions. Real-time walking without Web video encoding. Process time includes startup and teardown; loop time excludes setup/parse.'
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
