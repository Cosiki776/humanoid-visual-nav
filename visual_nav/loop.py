"""Bounded intent -> execute/observe -> evidence memory loop."""
import json
import math
import time
from pathlib import Path
from .geometry import horizontal_distance, path_length
from .memory import ObjectMemory
from .policy import validate_decision
from .task import COMMON_CATEGORIES
from .docking import arrival_geometry


class VisualNavigator:
    def __init__(self, io, detector, policy, output, mode='active_verify', max_observations=24, task=None, event_callback=None):
        self.io, self.detector, self.policy = io, detector, policy
        self.output, self.mode, self.limit = Path(output), mode, max_observations
        self.memory = ObjectMemory()
        self.events = []
        self.event_callback = event_callback
        self.current_observation = None
        self.task = task
        self.categories = tuple(task["detection_categories"]) if task else COMMON_CATEGORIES

    def notify(self, phase):
        if self.event_callback:
            self.event_callback(dict(type='live_state', phase=phase, task=self.task, arrival_geometry=getattr(self, 'proximity', None),
                                     observation=self.current_observation, memory=self.memory.summary(),
                                     events=self.events[-100:], observations=len(self.memory.observations),
                                     position=self.io.position.tolist(),
                                     search_coverage=self.io.search_summary()))

    def record(self, event):
        self.events.append(event)
        self.notify(event['type'])

    def observe(self, **kwargs):
        obs = self.io.observe(**kwargs)
        self.current_observation = dict(id=obs['id'], rgb_path=obs['rgb_path'])
        self.notify('perception')
        detections = self.detector.detect(obs, self.output, categories=self.categories)
        seen = self.memory.integrate(detections, obs)
        if self.events and self.events[-1]['type'] == 'execution':
            self.events[-1]['keyframes'] = [obs['id']]
            self.events[-1]['discoveries'] = seen
        self.record(dict(type='observation', id=obs['id'], seen=seen, position=obs['position']))
        print(json.dumps(self.events[-1]), flush=True)
        return obs

    def run(self, instruction):
        begin = time.monotonic()
        selected, goal, exploration = None, None, None
        status, verified = 'budget_exhausted', False
        initial_scan, inspect_count = 0, 0
        obs = None
        try:
            while len(self.memory.observations) < self.limit:
                kwargs = {'look_at': self.memory.instances[selected]['position']} if selected else {'yaw': initial_scan*math.pi/2}
                obs = self.observe(**kwargs)
                proximity = arrival_geometry(self.io.position, self.memory.instances[selected]) if selected else None
                self.proximity = proximity
                at_goal = bool(selected and goal is not None and horizontal_distance(self.io.position, goal) < .18
                               and proximity['close_enough'])
                # Initial-view ablation freezes perception after choosing and never asks for confirmation.
                if self.mode == 'candidates' and selected:
                    raise RuntimeError('Internal ablation control error')
                self.notify('thinking')
                decision = self.policy.decide(instruction, obs, self.memory, selected, at_goal)
                validate_decision(decision, self.memory.instances, obs['id'],
                                  self.memory.observations[-1]['seen'], selected, at_goal)
                self.record(dict(type='decision', observation_id=obs['id'], **decision))
                print(json.dumps(self.events[-1], ensure_ascii=False), flush=True)
                action = decision['action']
                if self.mode == 'verify' and at_goal and action not in {'confirm', 'reject'}:
                    status = 'unverified'
                    break
                if action == 'confirm':
                    if not proximity or not proximity['close_enough']:
                        raise ValueError('Cannot confirm arrival outside the target surface stopping range')
                    selected = decision['candidate_id']
                    status, verified = 'succeeded', True
                    break
                if action == 'reject':
                    self.memory.reject(selected, obs['id'], decision.get('reason', 'visual mismatch'))
                    selected, goal = None, None
                    inspect_count = 0
                    if self.mode != 'active_verify':
                        status = 'rejected_target'
                        break
                    continue
                if action == 'uncertain':
                    self.memory.questions.append(decision.get('reason', 'uncertain'))
                requested = decision.get('candidate_id') or (selected if action == 'observe' else None)
                if action in {'select', 'observe'} and requested:
                    new = requested
                    if new != selected:
                        selected, goal, inspect_count = new, None, 0
                        at_goal = False
                    options = self.io.docking_options(self.memory.instances[selected])
                    if not options:
                        self.memory.questions.append('No reachable docking point for '+selected)
                        selected, goal = None, None
                    else:
                        if at_goal and action in {'observe', 'select'}:
                            if self.mode != 'active_verify':
                                status = 'unverified'
                                break
                            inspect_count += 1
                            alternatives = [p for p in options if horizontal_distance(p, self.io.position) > .5]
                            goal = alternatives[min(inspect_count-1, len(alternatives)-1)] if alternatives else options[0]
                            if inspect_count > 3:
                                self.memory.questions.append('Inspection inconclusive for '+selected)
                                selected, goal = None, None
                        elif goal is None or not at_goal:
                            goal = options[0]
                elif action in {'explore', 'uncertain'}:
                    selected, goal = None, None
                if selected and goal is not None:
                    if self.mode in {'candidates', 'verify'}:
                        # Same controller and metric geometry; no subsequent detector or VLM calls.
                        for _ in range(80):
                            self.notify('moving')
                            event = self.io.advance(goal)
                            self.record(dict(type='execution', **event))
                            if event['status'] != 'succeeded' or event.get('reached'):
                                break
                        if self.mode == 'candidates' or not event.get('reached'):
                            status = ('arrived_unverified' if event.get('reached') else
                                      'timeout' if event['status'] == 'succeeded' else event['status'])
                            break
                        continue
                    self.notify('moving')
                    event = self.io.advance(goal)
                    self.record(dict(type='execution', **event))
                    if event['status'] != 'succeeded':
                        self.memory.questions.append('Movement failed; target validity unresolved')
                        selected, goal = None, None
                    continue
                # Four normal pinhole views at each search location; no panorama conversion needed.
                initial_scan += 1
                if initial_scan % 4:
                    continue
                exploration = self.io.explore_goal()
                if exploration is None:
                    coverage = self.io.search_summary()
                    status = 'search_complete_no_match' if coverage['complete'] else 'search_exhausted_unresolved'
                    break
                self.notify('moving')
                event = self.io.advance(exploration, budget=1.5)
                self.record(dict(type='execution', **event))
                if event['status'] != 'succeeded':
                    self.io.mark_search_failure(exploration)
                    exploration = None
        except Exception as exc:
            status = 'error'
            self.record(dict(type='error', error=type(exc).__name__+': '+str(exc)))
            raise
        finally:
            trace = self.io.trajectory()
            result = dict(status=status, mode=self.mode, policy=type(self.policy).__name__, instruction=instruction,
                          task=self.task, detection_categories=self.categories, selected=selected, verified=verified, start_position=self.io.start,
                          final_position=self.io.position.tolist(), observations=len(self.memory.observations),
                          elapsed_seconds=time.monotonic()-begin, path_length_m=path_length(trace),
                          vlm_calls=self.policy.calls, events=self.events, memory=self.memory.summary())
            result['search_coverage'] = getattr(self.io, 'search_summary', lambda: None)()
            if status in {'search_complete_no_match', 'search_exhausted_unresolved', 'budget_exhausted'}:
                result['message'] = {
                    'search_complete_no_match': '本轮可达搜索点已检查完，未发现符合条件的目标，停止搜索。',
                    'search_exhausted_unresolved': '可检查的搜索点已用尽，部分位置受阻或未核实，停止搜索。',
                    'budget_exhausted': '已达到本轮观察预算，尚未确认目标，停止搜索。'}[status]
            if selected:
                result['arrival_geometry'] = arrival_geometry(self.io.position, self.memory.instances[selected])
                result['selected_points'] = self.memory.instances[selected]['points']
            for filename, data in [('result.json', result), ('trajectory.json', trace)]:
                (self.output/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2))
            if hasattr(self.io, 'search_report'):
                (self.output/'search_coverage.json').write_text(json.dumps(self.io.search_report(), indent=2))
        return result
