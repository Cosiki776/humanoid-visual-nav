#!/usr/bin/env python3
"""Visual-only navigation entry point. No GT table, annotations, or GT session history."""
import argparse
import base64
import io as byteio
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
import bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observer-stream', action='store_true', help='Third-person movement frames for the Web only')
    p.add_argument('--realtime', action='store_true', help='Use real-time walking pace without emitting video frames')
    p.add_argument('--live-events', action='store_true', help='Emit visual-only Web state updates')
    p.add_argument('--scene', default='apt_0')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--start', type=float, nargs=3)
    p.add_argument('--instruction', default='走到白色椅子旁边')
    p.add_argument('--mode', choices=['capture', 'coverage', 'perception', 'candidates', 'verify', 'active_verify'], default='active_verify')
    p.add_argument('--policy', choices=['vlm', 'color'], default='vlm')
    p.add_argument('--vlm-model', help='Image-capable model; overrides NAV_VLM_MODEL')
    p.add_argument('--max-observations', type=int, default=24)
    p.add_argument('--travel-budget', type=float, default=3.0, help='Maximum metres between target approach observations (0.25..5, default 3)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--sam-checkpoint', default=str(bootstrap.WORKSPACE/'weights/sam_vit_b_01ec64.pth'))
    p.add_argument('--dino-model', default='IDEA-Research/grounding-dino-tiny')
    p.add_argument('--target-category', help='Explicit English target for perception or color-debug mode')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.vlm_model:
        os.environ['NAV_VLM_MODEL'] = args.vlm_model
    if not 1 <= args.max_observations <= 200:
        p.error('--max-observations must be 1..200')
    output = args.output or bootstrap.ROOT/'outputs'/('visual_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(vars(args), default=str, ensure_ascii=False, indent=2))
    setup_begin = time.monotonic()
    from visual_nav.habitat_io import HabitatIO
    def observer_frame(frame):
        from PIL import Image
        buffer = byteio.BytesIO()
        Image.fromarray(frame).save(buffer, format='JPEG', quality=78)
        jpeg = buffer.getvalue()
        temporary = output/'observer_last.tmp'
        temporary.write_bytes(jpeg)
        temporary.replace(output/'observer_last.jpg')
        print(json.dumps(dict(type='observer_frame', jpeg=base64.b64encode(jpeg).decode('ascii'))), flush=True)
    io = HabitatIO(args.scene, output, seed=args.seed, start=args.start,
                   frame_callback=observer_frame if args.observer_stream else None,
                   realtime=args.observer_stream or args.realtime)
    try:
        if args.mode == 'coverage':
            # Real cameras and navigation, no perception/VLM; only verifies search termination.
            status = 'budget_exhausted'
            for index in range(args.max_observations):
                io.observe(yaw=(index % 4)*math.pi/2)
                if index % 4 == 3:
                    goal = io.explore_goal()
                    if goal is None:
                        status = 'search_complete' if io.search_summary()['complete'] else 'search_exhausted'
                        break
                    event = io.advance(goal, budget=1.5)
                    if event['status'] != 'succeeded':
                        io.mark_search_failure(goal)
                    print(json.dumps(dict(io.search_summary(), movement=event)), flush=True)
            result = dict(status=status, mode='geometry_coverage_check', **io.search_summary())
            (output/'coverage.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
            (output/'search_coverage.json').write_text(json.dumps(io.search_report(), indent=2))
            print(json.dumps(result), flush=True)
        elif args.mode == 'capture':
            checks = []
            for index in range(min(args.max_observations, 8)):
                if index == 4:
                    goal = io.explore_goal()
                    if goal is not None:
                        print(io.advance(goal), flush=True)
                obs = io.observe(yaw=(index % 4)*math.pi/2)
                checks.append(dict(observation=obs['id'], **io.calibration_check(obs)))
                print(obs['id'], obs['position'], flush=True)
            (output/'trajectory.json').write_text(json.dumps(io.trajectory()))
            (output/'calibration.json').write_text(json.dumps(checks, indent=2))
        else:
            from visual_nav.perception import GroundedSAM
            from visual_nav.loop import VisualNavigator
            from visual_nav.policy import VLMPolicy, ColorPolicy
            detector = GroundedSAM(args.sam_checkpoint, args.dino_model, args.device)
            policy = VLMPolicy() if args.policy == 'vlm' and args.mode != 'perception' else ColorPolicy()
            if isinstance(policy, VLMPolicy):
                policy.trace_dir = output/"vlm_requests"
            from visual_nav.task import validate_task
            parse_begin = time.monotonic()
            if isinstance(policy, VLMPolicy):
                task = policy.parse_instruction(args.instruction)
            elif args.target_category:
                task = validate_task(dict(supported=True, target_categories=[args.target_category]))
                policy.task = task
            elif args.mode == 'perception':
                task = None
            else:
                p.error('--policy color requires --target-category; it cannot parse general instructions')
            parse_seconds = time.monotonic()-parse_begin
            print(json.dumps(dict(type='task', task=task, parse_seconds=parse_seconds), ensure_ascii=False), flush=True)
            (output/'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2))
            manifest = dict(detector.manifest, policy=type(policy).__name__,
                            vlm_model=getattr(policy, 'model', None), instruction_parse_seconds=parse_seconds, setup_seconds=time.monotonic()-setup_begin)
            (output/'models.json').write_text(json.dumps(manifest, indent=2))
            navigator = VisualNavigator(io, detector, policy, output, args.mode, args.max_observations, task=task, event_callback=(lambda event: print(json.dumps(event, ensure_ascii=False), flush=True)) if args.live_events else None, travel_budget=args.travel_budget)
            if args.mode == 'perception':
                for index in range(min(args.max_observations, 8)):
                    navigator.observe(yaw=index*math.pi/2)
                (output/'memory.json').write_text(json.dumps(navigator.memory.summary(), ensure_ascii=False, indent=2))
            else:
                result = navigator.run(args.instruction)
                print(json.dumps({k: result[k] for k in ('status', 'selected', 'observations', 'path_length_m', 'elapsed_seconds')}, ensure_ascii=False))
    except Exception as exc:
        # Structured failure survives errors before the navigation loop (e.g. parsing).
        failure = dict(type='run_error', error_type=type(exc).__name__, message=str(exc))
        (output/'error.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2))
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        raise
    finally:
        io.close()
    print('Output:', output)


if __name__ == '__main__':
    main()
