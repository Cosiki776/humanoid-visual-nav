"""Diagnose saved visual geometry and optionally run fresh navigation from that pose."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import bootstrap
from visual_nav.geometry import unproject
from visual_nav.habitat_io import HabitatIO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--navigate', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = json.loads((args.run/'result.json').read_text())
    config = json.loads((args.run/'config.json').read_text())
    instance = next(i for i in result['memory']['instances'] if i['id']==args.candidate)
    evidence = next(e for e in instance['evidence'] if e['observation_id']==args.observation)
    event = next(e for e in result['events'] if e['type']=='observation' and e['id']==args.observation)
    with np.load(args.run/(args.observation+'.npz')) as data:
        with Image.open(evidence['mask']) as source:
            points = unproject(data['depth'], np.asarray(source)>0, data['intrinsics'], data['camera_to_world'])
    lo, hi = np.quantile(points, [.05, .95], axis=0)
    visual = dict(id=args.candidate, position=np.median(points, axis=0).tolist(),
                  bbox_min=lo.tolist(), bbox_max=hi.tolist(), points=points.tolist())
    simulator = HabitatIO(config['scene'], args.output, start=event['position'], realtime=True)
    try:
        docks = simulator.docking_options(visual)
        diagnostics = dict(simulator.last_docking_diagnostics)
        approach = simulator.approach_options(visual, visited=[event['position']])
        report = dict(candidate=args.candidate, observation=args.observation, docking_diagnostics=diagnostics,
                      docking_options=[p.tolist() for p in docks], approach_options=[p.tolist() for p in approach])
        (args.output/'docking_diagnostics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        if args.navigate:
            from visual_nav.perception import GroundedSAM
            from visual_nav.policy import VLMPolicy
            from visual_nav.loop import VisualNavigator
            detector = GroundedSAM(str(bootstrap.WEIGHTS/'sam_vit_b_01ec64.pth'))
            policy = VLMPolicy()
            policy.trace_dir = args.output/'vlm_requests'
            task = policy.parse_instruction(result['instruction'])
            navigator = VisualNavigator(simulator, detector, policy, args.output, task=task, max_observations=24)
            fresh = navigator.run(result['instruction'])
            print(json.dumps({k:fresh[k] for k in ('status','selected','observations','elapsed_seconds')}, ensure_ascii=False))
    finally:
        simulator.close()


if __name__ == '__main__':
    main()
