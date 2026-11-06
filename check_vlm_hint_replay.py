"""Replay recorded RGB-D evidence through the current policy, without moving a robot."""
import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from visual_nav.geometry import unproject
from visual_nav.memory import ObjectMemory
from visual_nav.policy import VLMPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = json.loads((args.run/'result.json').read_text())
    event = next(e for e in saved['events'] if e['type']=='observation' and e['id']==args.observation)
    timestamp = next(e['time'] for i in saved['memory']['instances'] for e in i['evidence']
                     if e['observation_id'] == args.observation)
    frame = dict(id=args.observation, time=timestamp, seen=event['seen'])
    memory = ObjectMemory()
    for original in saved['memory']['instances']:
        evidence = [e for e in original['evidence'] if e['time'] <= frame['time']]
        if not evidence:
            continue
        latest = evidence[-1]
        directory = Path(latest['crop']).parent
        with np.load(directory/(latest['observation_id']+'.npz')) as data:
            with Image.open(latest['mask']) as source:
                points = unproject(data['depth'], np.asarray(source)>0, data['intrinsics'], data['camera_to_world'])
        if points is None:
            continue
        lo, hi = np.quantile(points, [.05, .95], axis=0)
        memory.instances[original['id']] = dict(id=original['id'], category=original['category'],
            evidence=copy.deepcopy(evidence), attributes=copy.deepcopy(latest['attributes']),
            confidence=latest['confidence'], views=len(evidence), last_seen=latest['observation_id'],
            position=np.median(points, axis=0).tolist(), bbox_min=lo.tolist(), bbox_max=hi.tolist(),
            points=points.tolist(), possible_same_ids=[], identity_state='replay')
    memory.observations = [frame]
    destination = args.output/(args.observation+'.jpg')
    shutil.copyfile(args.run/(args.observation+'.jpg'), destination)
    with np.load(args.run/(args.observation+'.npz')) as data:
        obs = dict(data)
    obs.update(id=args.observation, time=frame['time'], position=event['position'], rgb_path=str(destination))
    policy = VLMPolicy()
    policy.trace_dir = args.output/'vlm_requests'
    policy.task = copy.deepcopy(saved['task'])
    policy.task.pop('deferred_candidate_ids', None)
    decision = policy.decide(saved['instruction'], obs, memory)
    result = dict(decision=decision, calls=policy.calls, visual_input=policy.last_input,
                  limitations='Recorded frame replay; geometry reconstructed from past masks only. Not a navigation success test.')
    (args.output/'replay.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
