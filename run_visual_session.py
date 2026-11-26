#!/usr/bin/env python3
"""One persistent visual simulator/model owner; JSON commands arrive on stdin."""
import argparse
import base64
import io as byteio
import json
import os
import signal
import sys
import time
from pathlib import Path

import bootstrap
from visual_nav.memory import ObjectMemory
from visual_nav.task_context import bind_task, goal_snapshot


def emit(event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='apt_0')
    parser.add_argument('--session-dir', type=Path, required=True)
    parser.add_argument('--vlm-model', required=True)
    args = parser.parse_args()
    os.environ['NAV_VLM_MODEL'] = args.vlm_model
    args.session_dir.mkdir(parents=True, exist_ok=True)
    # Closing the parent pipe exits the owner; SIGTERM also releases simulator resources.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    simulator, detector = None, None
    memory = ObjectMemory()
    history = []
    previous_goal = None
    output = args.session_dir

    def frame(image):
        from PIL import Image
        buffer = byteio.BytesIO()
        Image.fromarray(image).save(buffer, format='JPEG', quality=78)
        jpeg = buffer.getvalue()
        temporary = output/'observer_last.tmp'
        temporary.write_bytes(jpeg)
        temporary.replace(output/'observer_last.jpg')
        emit(dict(type='observer_frame', jpeg=base64.b64encode(jpeg).decode('ascii')))

    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command.get('instruction') is None:
                # Preview only: no detector, model call, task history or movement.
                try:
                    output = args.session_dir
                    if simulator is None:
                        from visual_nav.habitat_io import HabitatIO
                        simulator = HabitatIO(args.scene, output, frame_callback=frame, realtime=True)
                    frame(simulator._world.frame(simulator.position))
                    emit(dict(type='session_ready', position=list(map(float, simulator.position))))
                except Exception as exc:
                    emit(dict(type='run_error', message=str(exc), error_type=type(exc).__name__))
                    raise
                continue
            output = Path(command['output'])
            output.mkdir(parents=True, exist_ok=False)
            stop = lambda: (args.session_dir/'cancel.request').exists()
            started = time.monotonic()
            result = None
            try:
                if simulator is None:
                    from visual_nav.habitat_io import HabitatIO
                    simulator = HabitatIO(args.scene, output, frame_callback=frame, realtime=True)
                simulator.begin_task(output, stop)
                # Re-publish the actual current pose, including after history playback.
                frame(simulator._world.frame(simulator.position))
                from visual_nav.directional import parse_directional, run_directional
                movement = parse_directional(command['instruction'])
                if movement:
                    emit(dict(type='task',task=movement,parse_seconds=0))
                    save_json(output/'task.json',movement)
                    save_json(output/'config.json',dict(command,session_id=args.session_dir.name,scene=args.scene))
                    result=run_directional(simulator,movement,command['instruction'],memory,emit)
                    save_json(output/'trajectory.json',simulator.trajectory())
                    continue
                if detector is None:
                    from visual_nav.perception import GroundedSAM
                    detector = GroundedSAM(str(bootstrap.WEIGHTS/'sam_vit_b_01ec64.pth'),
                                           'IDEA-Research/grounding-dino-tiny', 'cuda:0')
                from visual_nav.policy import VLMPolicy
                from visual_nav.loop import VisualNavigator
                policy = VLMPolicy()
                policy.trace_dir = output/"vlm_requests"
                parse_begin = time.monotonic()
                task = bind_task(policy.parse_instruction(command['instruction']), command['instruction'], previous_goal)
                policy.task = task
                parse_seconds = time.monotonic()-parse_begin
                emit(dict(type='task', task=task, parse_seconds=parse_seconds))
                save_json(output/'task.json', task)
                save_json(output/'models.json', dict(detector.manifest, policy='VLMPolicy',
                          vlm_model=args.vlm_model, instruction_parse_seconds=parse_seconds,
                          setup_seconds=time.monotonic()-started))
                save_json(output/'config.json', dict(command, session_id=args.session_dir.name, scene=args.scene))
                # Rejections/questions refer to the previous instruction, not permanent object truth.
                # Their full records remain in each run result and session history.
                memory.rejections.clear()
                memory.questions.clear()
                navigator = VisualNavigator(simulator, detector, policy, output,
                    max_observations=command['max_observations'], task=task,
                    event_callback=emit, memory=memory, should_stop=stop)
                result = navigator.run(command['instruction'])
                if result.get('verified') and result.get('status') == 'succeeded':
                    previous_goal = goal_snapshot(memory.instances[result['selected']])
            except Exception as exc:
                emit(dict(type='run_error', error_type=type(exc).__name__, message=str(exc)))
                saved = output/'result.json'
                result = json.loads(saved.read_text()) if saved.exists() else dict(
                    mode='active_verify', policy='VLMPolicy', observations=0, events=[],
                    start_position=simulator.start if simulator else None,
                    final_position=simulator.position.tolist() if simulator else None,
                    memory=memory.summary())
                result.update(status='cancelled' if stop() else 'error', error=str(exc),
                              instruction=command['instruction'])
            finally:
                if simulator is not None:
                    simulator._executor.stop()
                if result is not None:
                    result.update(session_id=args.session_dir.name, task_index=len(history)+1)
                    save_json(output/'result.json', result)
                    history.append(dict(run=output.name, instruction=command['instruction'],
                                        status=result['status'], selected=result.get('selected'),
                                        verified=result.get('verified', False), start_position=result.get('start_position'),
                                        final_position=simulator.position.tolist() if simulator else None))
                    save_json(args.session_dir/'legacy_gt.session.json', dict(session_id=args.session_dir.name,
                        scene=args.scene, tasks=history, previous_verified_goal=previous_goal,
                        position=simulator.position.tolist() if simulator else None,
                        yaw=simulator.yaw if simulator else None, observation_counter=simulator.counter if simulator else 0,
                        memory=memory.summary()))
                    save_json(args.session_dir/'visual_memory.json', dict(instances=memory.instances,
                              observations=memory.observations, relations=memory.relations.summary()))
                    emit(dict(type='task_done', run=output.name, status=result['status']))
    finally:
        if simulator is not None:
            simulator.close()


if __name__ == '__main__':
    main()
