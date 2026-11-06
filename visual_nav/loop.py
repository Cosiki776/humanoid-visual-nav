"""Bounded intent -> execute/observe -> evidence memory loop."""
import json
import math
import time
import os
from pathlib import Path
from .geometry import horizontal_distance, path_length
from .memory import ObjectMemory
from .policy import validate_decision, DecisionValidationError
from .task import COMMON_CATEGORIES
from .task_context import excluded_ids
from .docking import arrival_geometry
from .inspection import reference_for, joint_aim


def validate_target_change(decision, memory, task, selected, at_goal, program_control=False):
    action, key = decision['action'], decision.get('candidate_id')
    if action in {'select', 'observe', 'confirm'} and key in (task or {}).get('deferred_candidate_ids', []):
        raise DecisionValidationError('Candidate inspection budget exhausted for this task; search elsewhere', decision)
    if action in {'select', 'observe', 'confirm'} and key in excluded_ids(task, memory.instances):
        raise DecisionValidationError('This is the fixed previous goal, not another object', decision)
    if selected and at_goal and action in {'explore', 'uncertain'}:
        raise DecisionValidationError('At inspection pose: confirm, reject with current evidence, or observe; the excluded previous goal has not changed', decision)
    if selected and key and key != selected and action in {'select', 'observe'}:
        item = memory.instances[key]
        seen = memory.observations[-1]['seen']
        hint = decision.get('visual_hint', {})
        from .verification import grounded_hint
        explicit_hint = bool(program_control and isinstance(hint, dict) and
                             hint.get('observation_id') == memory.observations[-1]['id'] and
                             hint.get('candidate_id') == key and grounded_hint(
                                 {'target_hint': hint}, memory.observations[-1], memory, task))
        if not explicit_hint and (key not in seen or item['views'] < 2 or
                item['confidence'] < memory.instances[selected]['confidence']-.1 or
                not isinstance(decision.get('switch_reason'), str) or not decision['switch_reason'].strip()):
            raise DecisionValidationError('Target switch lacks repeated current evidence or switch_reason; re-observe the selected target instead', decision)
        if task and item['category'] not in task['target_categories']:
            raise DecisionValidationError('New target category does not match the task', decision)
    if action == 'reject' and (task or {}).get('another'):
        if decision.get('rejection_kind') not in {'category_mismatch', 'attribute_mismatch', 'relation_mismatch'}:
            raise DecisionValidationError('Reject needs rejection_kind for an observed mismatch; reaching a new target is not a reason to reject it', decision)
    if action == 'confirm' and (task or {}).get('relation'):
        reference = decision.get('reference_id')
        item = memory.instances.get(reference, {})
        current = memory.observations[-1]
        if program_control and task.get('relation') == 'near':
            from .near_navigation import relation_evidence
            if (reference != task.get('near_references', {}).get(key)
                    or item.get('category') not in task.get('reference_categories', [])
                    or decision.get('reference_verified') is not True
                    or not str(decision.get('reference_reason') or '').strip()
                    or not relation_evidence(memory, key, reference, current, final=True)):
                raise DecisionValidationError('Near confirmation needs verified bound reference and current or recent multiview geometry', decision)
            return
        if (reference == key or reference not in current['seen'] or
                item.get('category') not in task.get('reference_categories', []) or
                decision.get('reference_verified') is not True or
                not str(decision.get('reference_reason') or '').strip()):
            raise DecisionValidationError('Relation confirmation requires a currently visible reference_id, reference_verified=true and visual reference_reason; a cabinet is not a refrigerator', decision)
        supported = any(edge['relation'] == task['relation'] and
                        ((edge['source'] == key and edge['target'] == reference) or
                         (task['relation'] == 'near' and edge['source'] == reference and edge['target'] == key)) and
                        edge['status'] != 'contradicted' and edge['last_checked'] == current['id']
                        for edge in memory.relations.summary())
        if not supported:
            raise DecisionValidationError('Relation is not supported by current joint visual geometry; inspect or reject instead', decision)


class VisualNavigator:
    def __init__(self, io, detector, policy, output, mode='active_verify', max_observations=24, task=None, event_callback=None, travel_budget=3.0, memory=None, should_stop=None, execution_mode=None):
        if not math.isfinite(travel_budget) or not 0.25 <= travel_budget <= 5.0:
            raise ValueError('travel_budget must be 0.25..5.0 metres')
        self.execution_mode = execution_mode or os.environ.get('NAV_EXECUTION_MODE', 'sync')
        if self.execution_mode not in {'sync', 'async'}:
            raise ValueError('execution_mode must be sync or async')
        self.runtime = None
        self.perception_seconds = 4.0
        self.travel_budget = travel_budget
        self.io, self.detector, self.policy = io, detector, policy
        self.output, self.mode, self.limit = Path(output), mode, max_observations
        self.memory = memory if memory is not None else ObjectMemory()
        self.observation_offset = len(self.memory.observations)
        self.should_stop = should_stop or (lambda: False)
        self.events = []
        from queue import SimpleQueue
        self.request_events = SimpleQueue()
        if self.policy is not None:
            self.policy.request_callback = self.on_vlm_call
        self.event_callback = event_callback
        self.current_observation = None
        self.task = task
        self.program_control = bool(getattr(policy, 'program_control', False) and task and task.get('relation', '') in ('', 'near'))
        self.inspection_counts = {}
        self.deferred = set()
        self.inspection_positions = {}
        if task and task.get('another') and not task.get('excluded_goal'):
            raise ValueError('“另一个”需要会话中已确认目标作为固定参照')
        self.categories = tuple(task["detection_categories"]) if task else COMMON_CATEGORIES
        if self.program_control:
            self.categories = tuple(dict.fromkeys(task['target_categories'] + task.get('reference_categories', [])))

    def on_vlm_call(self, event):
        # Worker threads must not inspect simulator or mutable object memory.
        self.request_events.put(event)
        if self.event_callback:
            self.event_callback(dict(type='live_state', vlm_status=event))

    def notify(self, phase):
        while not self.request_events.empty():
            self.events.append(self.request_events.get())
        if self.event_callback:
            self.event_callback(dict(type='live_state', runtime_version='evidence-navigation-v3', execution_mode=self.execution_mode, phase=phase, task=self.task, arrival_geometry=getattr(self, 'proximity', None),
                                     excluded_candidate_ids=sorted(excluded_ids(self.task, self.memory.instances)),
                                     observation=self.current_observation, memory=self.memory.summary(), vlm_observability=True,
                                     events=self.events[-100:], observations=len(self.memory.observations)-self.observation_offset,
                                     position=self.io.position.tolist(),
                                     search_coverage=self.io.search_summary()))

    def defer_candidate(self, key):
        from .verification import defer_candidate
        defer_candidate(self.policy, self.memory, key)
        self.deferred = set(self.policy.deferred_candidate_ids)
        if self.task is not None:
            self.task['deferred_candidate_ids'] = sorted(self.deferred)

    def record(self, event):
        while not self.request_events.empty():
            self.events.append(self.request_events.get())
        self.events.append(event)
        self.notify(event['type'])

    def detection_overlay(self, obs, seen):
        """Snapshot frame-local boxes before rolling instance evidence changes."""
        boxes = []
        for key in seen:
            item = self.memory.instances[key]
            for evidence in item['evidence']:
                if evidence['observation_id'] == obs['id'] and evidence.get('bbox') is not None:
                    boxes.append(dict(id=key, category=item['category'],
                                      bbox=evidence['bbox'], confidence=evidence['confidence']))
        self.current_observation = dict(id=obs['id'], rgb_path=obs['rgb_path'], detections=boxes)
        return boxes

    def observe(self, **kwargs):
        started = time.perf_counter()
        obs = self.io.observe(**kwargs)
        captured = time.perf_counter()
        self.current_observation = dict(id=obs['id'], rgb_path=obs['rgb_path'])
        self.notify('perception')
        detections = self.detector.detect(obs, self.output, categories=self.categories)
        detected = time.perf_counter()
        self.perception_seconds = detected-captured
        seen = self.memory.integrate(detections, obs)
        boxes = self.detection_overlay(obs, seen)
        integrated = time.perf_counter()
        if self.events and self.events[-1]['type'] == 'execution':
            self.events[-1]['keyframes'] = [obs['id']]
            self.events[-1]['discoveries'] = seen
        self.record(dict(type='observation', id=obs['id'], seen=seen, position=obs['position'],
                         detections=boxes,
                         capture_seconds=captured-started, perception_seconds=detected-captured,
                         memory_seconds=integrated-detected))
        print(json.dumps(self.events[-1]), flush=True)
        return obs

    def advance(self, goal, instruction, selected=None):
        if self.execution_mode != 'async' or self.mode != 'active_verify':
            return self.io.advance(goal, budget=self.travel_budget)
        from .async_runtime import AsyncInference, semantic_signature, worthwhile_sample
        if self.runtime is None:
            self.runtime = AsyncInference(self.detector, self.policy, self.output, self.categories)
        runtime = self.runtime
        signature = semantic_signature(self.memory, self.task, selected)
        last_capture = self.io.position.copy()
        last_time = time.monotonic()
        missed = 0
        decision_source = None
        interrupt = False
        route = self.io.route(goal) if hasattr(self.io, 'route') else None
        remaining = min(12., path_length(route)) if route is not None else float('inf')
        previous_position = self.io.position.copy()

        def accept_observation(result, allow_decision=True):
            nonlocal signature, missed, decision_source
            obs, detections, elapsed = result
            self.perception_seconds = elapsed
            self.current_observation = dict(id=obs['id'], rgb_path=obs['rgb_path'])
            seen = self.memory.integrate(detections, obs)
            boxes = self.detection_overlay(obs, seen)
            self.record(dict(type='observation', id=obs['id'], seen=seen, position=obs['position'],
                             detections=boxes,
                             perception_seconds=elapsed, asynchronous=True))
            updated = semantic_signature(self.memory, self.task, selected)
            missed = missed+1 if selected and selected not in seen else 0
            changed = updated != signature or missed == 2
            if allow_decision and changed and runtime.reasoning is None and not self.should_stop() and not self.program_control:
                if runtime.submit_decision(instruction, obs, self.memory, selected):
                    decision_source = obs['id']
                    self.record(dict(type='async_decision_requested', observation_id=obs['id'],
                                     reason='new candidate, changed semantic evidence, or repeated target occlusion'))
                    signature = updated

        def tick():
            nonlocal last_capture, last_time, interrupt, remaining, previous_position
            remaining -= horizontal_distance(previous_position, self.io.position)
            previous_position = self.io.position.copy()
            if self.should_stop():
                return True
            result = runtime.take_observation()
            if result is not None:
                accept_observation(result)
            try:
                decision = runtime.take_decision()
            except DecisionValidationError as exc:
                self.record(dict(type='async_decision_invalid', error=str(exc)))
                decision = None
                interrupt = True
            if decision is not None:
                # Background suggestions never confirm arrival or mutate a target.
                # A changed intent interrupts motion and is revalidated on a fresh view.
                same = decision.get('action') == 'select' and decision.get('candidate_id') == selected
                interrupt = not same
                self.record(dict(type='async_decision', observation_id=decision_source,
                                 visual_input=getattr(self.policy, 'last_input', None),
                                 decision=decision, requires_fresh_review=interrupt))
            if interrupt:
                return True
            count = len(self.memory.observations)-self.observation_offset
            if runtime.perception is None and count < self.limit-1 and worthwhile_sample(remaining, self.perception_seconds) and (
                    horizontal_distance(last_capture, self.io.position) >= .8 and time.monotonic()-last_time >= .8):
                reference = reference_for(selected, self.task, self.memory) if selected else None
                if selected:
                    aim = joint_aim(self.memory.instances[selected], reference)
                else:
                    import numpy as np
                    aim = self.io.position+np.array([math.sin(self.io.yaw), 1.4, -math.cos(self.io.yaw)])
                obs = self.io.observe(look_at=aim)
                runtime.submit_observation(obs)
                last_capture, last_time = self.io.position.copy(), time.monotonic()
            return False

        # Continue only the already planned geometric route, with a bounded horizon.
        # Each controller step still checks NavMesh and cancellation.
        event = self.io.advance(goal, budget=12., on_step=tick)
        if self.should_stop():
            return event
        result = runtime.take_observation(wait=True)
        if result is not None:
            accept_observation(result, allow_decision=False)
        if runtime.reasoning is not None:
            try:
                decision = runtime.take_decision(wait=True)
                self.record(dict(type='async_decision', observation_id=decision_source, decision=decision,
                                 requires_fresh_review=True, reason='route ended; fresh stationary review follows'))
            except DecisionValidationError as exc:
                self.record(dict(type='async_decision_invalid', error=str(exc)))
        if event['status'] == 'interrupted':
            event['status'] = 'succeeded'
            event['visual_interrupt'] = True
        return event

    def run(self, instruction):
        begin = time.monotonic()
        selected, goal, exploration = None, None, None
        approaching = False
        approach_attempts = {}
        status, verified = 'budget_exhausted', False
        initial_scan, inspect_count = 0, 0
        invalid_decisions = 0
        transit_steps = 0
        inspection_frames = 0
        exploration_until_scan = None
        obs = None
        try:
            while len(self.memory.observations)-self.observation_offset < self.limit:
                if self.should_stop():
                    status = 'cancelled'
                    break
                search_yaw = None
                if hasattr(self.io, 'search_yaw'):
                    scans = self.io.search_summary()['completed_scans']
                    if exploration_until_scan is not None and scans >= exploration_until_scan:
                        exploration_until_scan = None
                        inspection_frames = 0
                    if inspection_frames >= 6 and exploration_until_scan is None and not self.program_control:
                        selected, goal = None, None
                        # Complete this area, then inspect another area before
                        # ambiguous candidates can interrupt exploration again.
                        exploration_until_scan = scans+2
                        invalid_decisions = 0
                        self.record(dict(type='search_resume', reason='Candidate inspection budget exhausted; inspect new search viewpoints',
                                         completed_scans=scans, resume_after_scans=exploration_until_scan))
                    if exploration_until_scan is not None:
                        selected, goal = None, None
                if not selected and hasattr(self.io, 'search_yaw'):
                    search_yaw = self.io.search_yaw()
                    if search_yaw is None:
                        exploration = self.io.explore_goal()
                        if exploration is None:
                            coverage = self.io.search_summary()
                            status = 'search_complete_no_match' if coverage['complete'] else 'search_exhausted_unresolved'
                            break
                        # Covered corridors need transit, not another four model calls.
                        # Bound transit independently of the observation budget.
                        if transit_steps >= 40:
                            status = 'search_exhausted_unresolved'
                            self.memory.questions.append('Search transit budget exhausted without a new observation location')
                            break
                        before = self.io.position.copy()
                        self.notify('moving')
                        event = self.advance(exploration, instruction)
                        self.record(dict(type='execution', purpose='search', destination=exploration.tolist(), **event))
                        transit_steps += 1
                        if event['status'] != 'succeeded' or horizontal_distance(before, self.io.position) < .05:
                            self.io.mark_search_failure(exploration)
                        continue
                reference = reference_for(selected, self.task, self.memory) if selected else None
                kwargs = {'look_at': joint_aim(self.memory.instances[selected], reference)} if selected else {'yaw': search_yaw if search_yaw is not None else initial_scan*math.pi/2}
                obs = self.observe(**kwargs)
                if selected and selected in excluded_ids(self.task, self.memory.instances):
                    self.record(dict(type='identity_excluded', candidate_id=selected,
                                     reason='New evidence links candidate to previous goal'))
                    selected, goal = None, None
                if selected:
                    self.inspection_positions.setdefault(selected, []).append(self.io.position.tolist())
                    inspection_frames += 1
                    self.inspection_counts[selected] = self.inspection_counts.get(selected, 0)+1
                    if self.inspection_counts[selected] >= 6:
                        self.defer_candidate(selected)
                        self.memory.questions.append('Inspection budget exhausted for '+selected+'; not evidence of absence')
                        self.record(dict(type='candidate_deferred', candidate_id=selected))
                        selected, goal = None, None
                        continue
                transit_steps = 0
                if self.should_stop():
                    status = 'cancelled'
                    break
                proximity = arrival_geometry(self.io.position, self.memory.instances[selected]) if selected else None
                self.proximity = proximity
                at_goal = bool(selected and not approaching and goal is not None and horizontal_distance(self.io.position, goal) < .18
                               and proximity['close_enough'])
                # Initial-view ablation freezes perception after choosing and never asks for confirmation.
                if self.mode == 'candidates' and selected:
                    raise RuntimeError('Internal ablation control error')
                self.notify('thinking')
                try:
                    decision = (dict(action='explore', candidate_id=None,
                                     reason='Bounded candidate inspection: continuing geometric search with visual memory updates')
                                if exploration_until_scan is not None else
                                self.policy.decide(instruction, obs, self.memory, selected, at_goal))
                    if self.program_control:
                        active_deferred = set(getattr(self.policy, 'deferred_candidate_ids', []))
                        for key in self.deferred - active_deferred:
                            self.inspection_counts.pop(key, None)
                            self.record(dict(type='candidate_reactivated', candidate_id=key,
                                             observation_id=obs['id'], reason='New viewpoint or larger visible extent'))
                        self.deferred = active_deferred
                        self.task['deferred_candidate_ids'] = sorted(self.deferred)
                    if self.program_control and decision['action'] == 'defer':
                        key = decision['candidate_id']
                        self.defer_candidate(key)
                        self.record(dict(type='candidate_deferred', candidate_id=key, reason=decision['reason']))
                        selected, goal, inspection_frames = None, None, 0
                        continue
                    validate_decision(decision, self.memory.instances, obs['id'],
                                      self.memory.observations[-1]['seen'], selected, at_goal)
                    validate_target_change(decision, self.memory, self.task, selected, at_goal, self.program_control)
                except DecisionValidationError as exc:
                    invalid_decisions += 1
                    feedback = dict(observation_id=obs['id'], selected=selected, error=str(exc))
                    self.policy.decision_feedback = (getattr(self.policy, 'decision_feedback', [])+[feedback])[-3:]
                    self.record(dict(type='decision_retry', proposed_decision=exc.decision,
                                     attempt=invalid_decisions, **feedback))
                    if invalid_decisions >= 3:
                        status = 'decision_unresolved'
                        break
                    # No movement or target change is authorized by an invalid decision.
                    # Reacquire an actual observation; normal observation budget still applies.
                    continue
                if self.should_stop():
                    status = 'cancelled'
                    break
                invalid_decisions = 0
                self.record(dict(type='decision', observation_id=obs['id'], visual_input=getattr(self.policy, 'last_input', None), **decision))
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
                        approaching = False
                        self.inspection_positions.setdefault(selected, []).append(self.io.position.tolist())
                        at_goal = False
                    options = self.io.docking_options(self.memory.instances[selected])
                    self.record(dict(type='docking_plan', candidate_id=selected, options=len(options),
                                     diagnostics=dict(getattr(self.io, 'last_docking_diagnostics', {}))))
                    if options and approaching:
                        goal, approaching = options[0], False
                    if action == 'observe' and hasattr(self.io, 'inspection_options'):
                        reference = reference_for(selected, self.task, self.memory)
                        ids = [selected] + ([reference['id']] if reference else [])
                        views = self.memory.views.retrieve(ids, limit=8)
                        inspections = self.io.inspection_options(self.memory.instances[selected], reference, views,
                            self.inspection_positions.get(selected, []))
                        if inspections:
                            goal = inspections[0]
                            approaching = False
                            self.record(dict(type='inspection_plan', candidate_id=selected,
                                reference_id=reference['id'] if reference else None, destination=goal.tolist(),
                                reason='Reachable estimated joint-view geometry; visibility still needs actual observation'))
                            options = inspections
                            at_goal = False
                        else:
                            self.memory.questions.append('No new estimated joint viewpoint for '+selected)
                            self.record(dict(type='approach_blocked', candidate_id=selected,
                                             reason='No new reachable inspection viewpoint'))
                            self.defer_candidate(selected)
                            selected, goal = None, None
                            continue
                    if not options and action == 'select' and self.mode == 'active_verify' and hasattr(self.io, 'approach_options'):
                        if approaching and goal is not None and horizontal_distance(self.io.position, goal) >= .18:
                            options = [goal]
                        elif approach_attempts.get(selected, 0) < 3:
                            options = self.io.approach_options(self.memory.instances[selected],
                                                              self.inspection_positions.get(selected, []))
                            if options:
                                goal, approaching = options[0], True
                                approach_attempts[selected] = approach_attempts.get(selected, 0)+1
                                self.record(dict(type='approach_plan', candidate_id=selected,
                                                 destination=goal.tolist(), attempt=approach_attempts[selected],
                                                 reason='No final docking point; approach for fresh geometry, not arrival'))
                        if not options:
                            status = 'target_approach_unresolved'
                            self.record(dict(type='approach_blocked', candidate_id=selected,
                                             attempts=approach_attempts.get(selected, 0),
                                             reason='No new reachable approach viewpoint or approach budget exhausted'))
                            break
                    if not options:
                        self.memory.questions.append('No reachable docking point for '+selected)
                        self.record(dict(type='approach_blocked', candidate_id=selected,
                                         reason='No reachable docking point'))
                        self.defer_candidate(selected)
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
                        elif goal is None or (action != 'observe' and not arrival_geometry(goal, self.memory.instances[selected])['close_enough']):
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
                    self.record(dict(type='target_motion', candidate_id=selected,
                                     reason='Controller requested to approach target'))
                    event = self.advance(goal, instruction, selected)
                    self.record(dict(type='execution', candidate_id=selected, **event))
                    if event['status'] != 'succeeded':
                        self.memory.questions.append('Movement failed; target validity unresolved')
                        selected, goal = None, None
                    continue
                # Four normal pinhole views at each search location; no panorama conversion needed.
                if hasattr(self.io, 'search_yaw'):
                    continue
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
            if hasattr(self.io, 'settle'):
                self.io.settle()
            if self.runtime is not None:
                self.runtime.close()
            while not self.request_events.empty():
                self.events.append(self.request_events.get())
            if self.should_stop():
                status, verified = 'cancelled', False
            trace = self.io.trajectory()
            result = dict(status=status, execution_mode=self.execution_mode, runtime_version='evidence-navigation-v3', mode=self.mode, policy=type(self.policy).__name__, instruction=instruction,
                          decision_mode='program' if self.program_control else 'legacy',
                          task=self.task, detection_categories=self.categories, selected=selected, verified=verified, start_position=self.io.start,
                          final_position=self.io.position.tolist(), observations=len(self.memory.observations)-self.observation_offset,
                          elapsed_seconds=time.monotonic()-begin, path_length_m=path_length(trace),
                          vlm_calls=self.policy.calls, vlm_timings=getattr(self.policy, 'metrics', []), travel_budget_m=self.travel_budget,
                          events=self.events, memory=self.memory.summary())
            result['search_coverage'] = getattr(self.io, 'search_summary', lambda: None)()
            result['excluded_candidate_ids'] = sorted(excluded_ids(self.task, self.memory.instances))
            if status == 'decision_unresolved':
                result['message'] = '连续三次决策缺少有效证据，机器人保持当前位置，目标尚未核实。'
            if status == 'target_approach_unresolved':
                result['message'] = '已发现目标候选，但未找到满足停靠条件的位置；接近观察恢复未能解决，未确认到达。'
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
