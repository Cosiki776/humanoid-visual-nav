#!/usr/bin/env python3
"""Replay identical visual inputs on cloud and local models, without moving the robot."""
import argparse
import json
import statistics
from pathlib import Path
import bootstrap
from model_settings import load_model_settings
from visual_nav.policy import VLMPolicy, validate_decision
from visual_nav.memory import ObjectMemory


def from_result(path):
    """An offline final-memory snapshot, NOT a historical online decision replay."""
    result = json.loads(path.read_text())
    obs_event = next(e for e in reversed(result['events']) if e['type']=='observation')
    obs = dict(id=obs_event['id'], position=obs_event['position'],
               rgb_path=str(path.parent/(obs_event['id']+'.jpg')))
    memory = ObjectMemory()
    memory.instances = {i['id']:i for i in result['memory']['instances']}
    memory.observations = [dict(id=obs['id'], seen=obs_event['seen'])]
    memory.rejections = result['memory'].get('rejections', [])
    memory.questions = result['memory'].get('unresolved', [])
    memory.relations.edges = {(e['source'],e['relation'],e['target']):e for e in result['memory'].get('relations',[])}
    policy = VLMPolicy()
    policy.task = result['task']
    captured = {}
    def capture(system, content):
        captured.update(system=system, content=content,
                        source='reconstructed final-memory snapshot; not exact historical replay')
        return dict(action='explore', candidate_id=None)
    policy._request = capture
    policy.decide(result['instruction'], obs, memory, selected=None, at_goal=False)
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--request', type=Path, help='Exact recorded vlm_requests/request_XXXX.json')
    source.add_argument('--result', type=Path, help='Build offline snapshot from an old result.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-base', default='http://127.0.0.1:11434')
    parser.add_argument('--local-model', default='qwen3.8:27b-q4_K_M')
    parser.add_argument('--cloud-model', default='qwen3.8-max')
    parser.add_argument('--num-ctx', type=int, default=16384)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if args.repeats < 1: parser.error('--repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_model_settings()
    # Constructor requires an explicit visual model; endpoints are set below.
    import os
    os.environ.setdefault('NAV_VLM_MODEL', args.cloud_model)
    sample = json.loads(args.request.read_text()) if args.request else from_result(args.result)
    (args.output/'input.json').write_text(json.dumps(sample, ensure_ascii=False))
    rows = []
    for repetition in range(args.repeats):
        # Alternate order to reduce systematic order bias. First calls may include loading.
        for backend in (['cloud','local'] if repetition%2==0 else ['local','cloud']):
            policy = VLMPolicy()
            policy.base = config['base_url'] if backend=='cloud' else args.local_base.rstrip('/')
            policy.model = args.cloud_model if backend=='cloud' else args.local_model
            policy.backend = 'openai' if backend=='cloud' else 'ollama'
            policy.key = config['api_key'] if backend=='cloud' else ''
            policy.num_ctx, policy.timeout = args.num_ctx, args.timeout
            policy.trace_dir = args.output/('%s_%02d' % (backend,repetition+1))
            row = dict(backend=backend, repetition=repetition+1, decision_valid=False)
            try:
                decision = policy._request(sample['system'],sample['content'])
                row['decision'] = decision
                content = sample['content']
                if isinstance(content, list):
                    context = json.loads(content[0]['text'])
                    validate_decision(decision, {i['id'] for i in context['candidates']},
                        context['observation_id'], context['currently_seen'], context['selected'], context['at_goal'])
                    row['decision_valid'] = True
            except Exception as exc:
                row['error'] = type(exc).__name__+': '+str(exc)
            row['timing'] = policy.metrics[-1] if policy.metrics else {}
            rows.append(row)
            (args.output/'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {}
    for backend in ['cloud','local']:
        group = [r for r in rows if r['backend']==backend]
        successful = [r['timing']['wall_seconds'] for r in group if r['timing'].get('ok')]
        summary[backend] = dict(requests=len(group), json_successes=len(successful),
            valid_decisions=sum(r['decision_valid'] for r in group),
            median_successful_seconds=statistics.median(successful) if successful else None,
            first_request_seconds=group[0]['timing'].get('wall_seconds'),
            subsequent_seconds=[r['timing'].get('wall_seconds') for r in group[1:]])
    summary['limitations'] = 'Same-input inference latency only; repeats may benefit from prefix/image caching. Validation is action schema/evidence consistency, not semantic accuracy or navigation success. First request is not guaranteed cold.'
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
