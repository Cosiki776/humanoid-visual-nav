#!/usr/bin/env python3
"""Evaluator-owned subprocess harness. Pass only public episode fields to agents."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from evaluate_visual import aggregate, score_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modes', nargs='+', choices=['gt', 'candidates', 'verify', 'active_verify'],
                   default=['gt', 'candidates', 'verify', 'active_verify'])
    p.add_argument('--policy', choices=['vlm', 'color'], default='vlm')
    p.add_argument('--vlm-model', default=os.environ.get('NAV_VLM_MODEL'))
    p.add_argument('--max-observations', type=int, default=24)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    scenes = {}
    episodes = json.loads(args.episodes.read_text())
    for episode_index, episode in enumerate(episodes):
        scene = episode['scene']
        if scene not in scenes:
            gt_path = args.output/('private_gt_%02d.json' % len(scenes))
            with (args.output/('export_%02d.log' % len(scenes))).open('w') as log:
                subprocess.run([sys.executable, str(root/'evaluate_visual.py'), '--export-scene', scene,
                                '--output', str(gt_path)], stdout=log, stderr=subprocess.STDOUT, check=True)
            scenes[scene] = json.loads(gt_path.read_text())['objects']
        starts = []
        for mode in args.modes:
            out = args.output/('%03d_%s' % (episode_index, mode))
            script = 'run_gt_baseline.py' if mode == 'gt' else 'run_visual.py'
            command = [sys.executable, str(root/script), '--scene', scene, '--seed', str(episode.get('seed', 7)),
                       '--instruction', episode['instruction'], '--output', str(out)]
            if 'start' in episode:
                command.extend(['--start']+[str(v) for v in episode['start']])
            if mode != 'gt':
                command += ['--mode', mode, '--policy', args.policy, '--max-observations', str(args.max_observations)]
                if args.vlm_model:
                    command += ['--vlm-model', args.vlm_model]
            # No goal_ids/annotations/GT path/episode filename is sent to visual subprocesses.
            with (args.output/('%03d_%s.log' % (episode_index, mode))).open('w') as log:
                process = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            if not (out/'result.json').exists():
                raise RuntimeError('Run did not produce a result; inspect '+str(out)+'. No fabricated metrics.')
            result = json.loads((out/'result.json').read_text())
            starts.append(result['start_position'])
            if any(abs(a-b) > .02 for a, b in zip(starts[0], starts[-1])):
                raise RuntimeError('Ablations have inconsistent starting poses')
            row = score_episode(result, episode, scenes[scene])
            row['process_returncode'] = process.returncode
            rows.append(row)
            (args.output/'scores.json').write_text(json.dumps({'episodes': rows, 'aggregate': aggregate(rows)}, ensure_ascii=False, indent=2))
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
