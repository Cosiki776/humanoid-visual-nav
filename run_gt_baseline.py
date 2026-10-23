#!/usr/bin/env python3
"""Original GT planner/session, with a geometry-only seeded start for fair comparison."""
import argparse
import json
import time
from pathlib import Path
import bootstrap
from world import World
from executor import Executor
from session import NavigationSession
from planning import ModelPlanner
from visual_nav.geometry import path_length


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='apt_0')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--start', nargs=3, type=float)
    p.add_argument('--instruction', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(vars(args), default=str, ensure_ascii=False))
    begin = time.monotonic()
    world = World(args.scene, seed=args.seed)
    try:
        start = args.start if args.start is not None else world.pf.get_random_navigable_point(island_index=world.island)
        # Existing annotations are apt_0 only; never reuse handle-based colors in another scene.
        attrs = json.loads((bootstrap.ROOT/'annotations.json').read_text()) if args.scene == 'apt_0' else {}
        scene = world.build_map(attrs)
        executor = Executor(world, start)
        setup_seconds = time.monotonic()-begin
        begin = time.monotonic()
        result = NavigationSession(world, executor, scene, ModelPlanner()).run_turn(args.instruction)
        result.update(mode='gt', policy='ModelPlanner', verified=False, observations=0,
                      path_length_m=path_length(executor.trace), elapsed_seconds=time.monotonic()-begin,
                      setup_seconds=setup_seconds)
        (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (args.output/'trajectory.json').write_text(json.dumps(executor.trace))
        print(result['status'])
    finally:
        world.close()


if __name__ == '__main__':
    main()
