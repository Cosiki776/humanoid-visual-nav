"""Run real sequential instructions, then independently score saved results.

No ground truth is passed to the navigation worker. Output must be a new path.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='qwen3.8-max')
    parser.add_argument('--max-observations', type=int, default=24)
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.score_only:
        rows = []
        for target in sorted(output.glob('task_*/result.json')):
            result = json.loads(target.read_text())
            row = {key: result.get(key) for key in ('status', 'selected', 'verified', 'observations',
                   'path_length_m', 'elapsed_seconds', 'vlm_calls', 'instruction')}
            row['output'] = str(target.parent)
            rows.append(row)
        score_results(output, rows)
        return
    output.mkdir(parents=True, exist_ok=False)
    instructions = ['走到白色椅子旁边', '走到另一把白色椅子旁边', '找柜子']
    rows = []
    with (output/'worker_stderr.log').open('w') as errors:
        worker = subprocess.Popen([sys.executable, 'run_visual_session.py', '--scene', 'apt_0',
                                   '--session-dir', str(output/'session'), '--vlm-model', args.model],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            for index, instruction in enumerate(instructions):
                target = output/('task_%02d' % (index+1))
                command = dict(instruction=instruction, output=str(target), max_observations=args.max_observations)
                worker.stdin.write(json.dumps(command, ensure_ascii=False)+'\n')
                worker.stdin.flush()
                for line in worker.stdout:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get('type') in {'task', 'run_error', 'task_done'}:
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                    if event.get('type') == 'task_done':
                        result = json.loads((target/'result.json').read_text())
                        row = {key: result.get(key) for key in ('status', 'selected', 'verified',
                               'observations', 'path_length_m', 'elapsed_seconds', 'vlm_calls')}
                        row.update(instruction=instruction, output=str(target))
                        rows.append(row)
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                        break
                else:
                    raise RuntimeError('Navigation worker exited before task completion')
        finally:
            worker.stdin.close()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait(timeout=15)
    score_results(output, rows)


def score_results(output, rows):
    # Offline scoring begins only after the navigation process has exited.
    from evaluate_visual import score_episode
    gt = json.loads(Path('outputs/evaluation/apt_0_gt.json').read_text())
    white = json.loads(Path('outputs/evaluation/white_episode.json').read_text())[0]
    previous = None
    for index, row in enumerate(rows):
        result = json.loads((Path(row['output'])/'result.json').read_text())
        if index < 2:
            goals = [key for key in white['goal_ids'] if index == 0 or key != previous]
        else:
            # ReplicaCAD uses wall_cabinet, not the generic detector noun.
            goals = [obj['id'] for obj in gt['objects'] if obj['category'] in {'cabinet', 'wall_cabinet'}]
        spec = dict(id='program_%d' % index, goal_ids=goals, min_target_gap_m=.35, max_target_gap_m=.65)
        score = score_episode(result, spec, gt['objects'])
        if index == 0:
            previous = score['matched_instance']
        row['offline_score'] = score
    (output/'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
