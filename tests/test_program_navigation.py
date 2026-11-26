"""Behavioral regressions for fixed-flow visual navigation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from tests.test_visual_nav import detection, observation
from visual_nav.memory import ObjectMemory
from visual_nav.loop import VisualNavigator
from visual_nav.policy import VLMPolicy, DecisionValidationError
from visual_nav.task import simple_navigation_task
from visual_nav.task_context import bind_task, goal_snapshot, excluded_ids


class ProgramNavigationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name)/'image.jpg')
        Image.new('RGB', (64, 64), 'white').save(self.path)
        self.memory = ObjectMemory()
        self.policy = object.__new__(VLMPolicy)
        self.policy.program_control = True
        self.policy.visual_checks = {}
        self.policy.task = simple_navigation_task('去白色椅子旁边')
        self.policy._request = Mock(side_effect=self.answer)
        self.verdict = 'yes'

    def answer(self, system, content):
        context = json.loads(content[0]['text'])
        return dict(candidate_id=context['candidate_id'], observation_id=context['observation_id'],
                    category_match='yes', attributes_match=self.verdict, reason='Visible evidence')

    def observe(self, key, offsets=(0,)):
        obs = observation(key)
        obs['rgb_path'] = self.path
        detections = [detection(offset) for offset in offsets]
        for d in detections:
            d.update(crop=self.path, bbox=[10, 10, 50, 50])
        self.memory.integrate(detections, obs)
        return obs

    def test_approach_reuses_check_but_arrival_requires_new_visual_evidence(self):
        obs = self.observe('o1')
        self.assertEqual(self.policy.decide('white chair', obs, self.memory)['action'], 'select')
        obs = self.observe('o2')
        self.assertEqual(self.policy.decide('', obs, self.memory, 'v0001')['action'], 'select')
        self.assertEqual(self.policy._request.call_count, 1)
        self.verdict = 'unknown'
        self.assertEqual(self.policy.decide('', obs, self.memory, 'v0001', True)['action'], 'observe')
        self.assertEqual(self.policy._request.call_count, 2)
        self.verdict = 'yes'
        obs = self.observe('o3')
        self.assertEqual(self.policy.decide('', obs, self.memory, 'v0001', True)['action'], 'confirm')

    def test_another_excludes_previous_before_model_and_uses_history(self):
        self.observe('o1', (0, 3))
        self.policy.task = bind_task(self.policy.task, '去另一把白色椅子旁边',
                                    goal_snapshot(self.memory.instances['v0001']))
        obs = self.observe('o2', (0, 3))
        result = self.policy.decide('', obs, self.memory)
        self.assertEqual(result['candidate_id'], 'v0002')
        self.assertEqual(self.policy.last_input['image_count'], 3)

    def test_missing_target_gets_new_view_without_model_call(self):
        self.observe('o1')
        obs = self.observe('o2', ())
        self.assertEqual(self.policy.decide('', obs, self.memory, 'v0001', True)['action'], 'observe')
        self.policy._request.assert_not_called()
        obs = self.observe('o3', ())
        self.assertEqual(self.policy.decide('', obs, self.memory, 'v0001', True)['action'], 'defer')
        self.policy._request.assert_not_called()

    def test_negative_candidate_is_not_repeatedly_queried_at_same_position(self):
        self.verdict = 'no'
        obs = self.observe('o1')
        self.assertEqual(self.policy.decide('', obs, self.memory)['action'], 'explore')
        obs = self.observe('o2')
        self.assertEqual(self.policy.decide('', obs, self.memory)['action'], 'explore')
        self.assertEqual(self.policy._request.call_count, 1)

    def test_wrong_evidence_id_cannot_confirm(self):
        obs = self.observe('o1')
        self.policy._request.side_effect = None
        self.policy._request.return_value = dict(candidate_id='v0001', observation_id='stale',
                                                category_match='yes', attributes_match='yes')
        with self.assertRaises(DecisionValidationError):
            self.policy.decide('', obs, self.memory, 'v0001', True)

    def test_ambiguous_fragment_excluded_across_turns(self):
        self.observe('o1')
        self.memory.instances['fragment'] = dict(self.memory.instances['v0001'], id='fragment',
                                                position=[2, 0, 0], possible_same_ids=['v0001'])
        task = bind_task(self.policy.task, '去另一把白色椅子旁边',
                         goal_snapshot(self.memory.instances['v0001']))
        self.assertIn('fragment', excluded_ids(task, self.memory.instances))

    def test_surface_overlap_handles_shifted_median(self):
        # Shared surfaces dominate geometrically; unequal sampling moves medians.
        rng = np.random.RandomState(2)
        a = rng.uniform([0, 0, -2], [.1, .1, -1.9], (100, 3))
        b = a + [.7, 0, 0]
        first, second = detection(), detection()
        first['points'] = np.concatenate([a, a, b])
        second['points'] = np.concatenate([a, b, b])
        self.memory.integrate([first], observation('o1'))
        self.assertEqual(self.memory.integrate([second], observation('o2')), ['v0001'])

    def test_reprojection_checks_depth_before_using_mask_overlap(self):
        self.observe('o1')
        obs = observation('o2')
        obs.update(depth=np.full((64, 64), 2.), intrinsics=np.array([[20,0,32],[0,20,32],[0,0,1]]))
        item = self.memory.instances['v0001']
        self.assertEqual(self.memory.projected_support(item, np.ones((64,64), bool), obs), 1.)
        self.assertEqual(self.memory.projected_support(item, np.zeros((64,64), bool), obs), 0.)
        obs['depth'][:] = .5
        self.assertIsNone(self.memory.projected_support(item, np.ones((64,64), bool), obs))

    def test_color_only_ranks_candidates_never_confirms_them(self):
        obs = self.observe('o1', (0, 3))
        self.memory.instances['v0002']['attributes'] = {'color_scores': {'white': 1.}}
        self.verdict = 'no'
        result = self.policy.decide('', obs, self.memory)
        context = json.loads(self.policy._request.call_args_list[0].args[1][0]['text'])
        self.assertEqual(context['candidate_id'], 'v0002')
        self.assertEqual(result['action'], 'explore')

    def test_unknown_first_candidate_does_not_hide_second_match(self):
        obs = self.observe('o1', (0, 3))
        def answer(system, content):
            value = self.answer(system, content)
            if value['candidate_id'] == 'v0001':
                value['category_match'] = 'unknown'
            return value
        self.policy._request.side_effect = answer
        result = self.policy.decide('', obs, self.memory)
        self.assertEqual(result['candidate_id'], 'v0002')
        self.assertEqual(result['action'], 'select')

    def test_explicit_hint_overrides_low_detector_rank_and_switch_gate(self):
        from visual_nav.loop import validate_target_change
        obs = self.observe('o1', (0, 3))
        self.memory.instances['v0002']['confidence'] = .1
        def answer(system, content):
            value = self.answer(system, content)
            value.update(category_match='unknown', attributes_match='unknown', target_hint=dict(
                candidate_id='v0002', category_match='yes', attributes_match='yes',
                reason='White chair to the left of the refrigerator'))
            return value
        self.policy._request.side_effect = answer
        result = self.policy.decide('', obs, self.memory, 'v0001')
        self.assertEqual(result['candidate_id'], 'v0002')
        validate_target_change(result, self.memory, self.policy.task, 'v0001', False, True)
        self.assertFalse(self.policy.visual_checks['v0002']['matches'])
        self.verdict = 'unknown'
        self.policy._request.side_effect = self.answer
        obs = self.observe('o2', (0, 3))
        self.assertEqual(self.policy.decide('', obs, self.memory, 'v0002', True)['action'], 'observe')

    def test_hint_cannot_override_previous_goal_exclusion(self):
        from visual_nav.verification import grounded_hint
        obs = self.observe('o1', (0, 3))
        task = bind_task(self.policy.task, '去另一把白色椅子旁边', goal_snapshot(self.memory.instances['v0001']))
        answer = dict(target_hint=dict(candidate_id='v0001', category_match='yes', attributes_match='yes', reason='white chair'))
        self.assertIsNone(grounded_hint(answer, obs, self.memory, task))
        answer['target_hint']['candidate_id'] = 'invented'
        self.assertIsNone(grounded_hint(answer, obs, self.memory, task))

    def test_approach_recommendation_is_not_arrival_confirmation(self):
        obs = self.observe('o1')
        self.verdict = 'unknown'
        def answer(system, content):
            return dict(self.answer(system, content), approach_recommended=True)
        self.policy._request.side_effect = answer
        self.assertEqual(self.policy.decide('', obs, self.memory)['action'], 'select')
        obs = self.observe('o2')
        self.assertEqual(self.policy.decide('', obs, self.memory, 'v0001', True)['action'], 'observe')

    def test_deferred_candidate_needs_new_information_to_reactivate(self):
        from visual_nav.verification import defer_candidate, recover_candidate
        self.observe('o1')
        self.policy.inspection_attempts = {}
        defer_candidate(self.policy, self.memory, 'v0001')
        self.observe('o2')
        self.assertFalse(recover_candidate(self.policy, self.memory, 'v0001'))
        self.memory.instances['v0001']['evidence'][-1]['bbox'] = [0, 0, 64, 64]
        self.assertTrue(recover_candidate(self.policy, self.memory, 'v0001'))
        self.assertNotIn('v0001', self.policy.deferred_candidate_ids)

    def test_deferred_candidate_does_not_disable_later_candidates(self):
        class IO:
            start = [0, 0, 0]
            position = np.zeros(3)
            count = 0
            def observe(self, **kwargs):
                self.count += 1
                return observation('o'+str(self.count))
            def trajectory(self): return [self.start, self.position.tolist()]
            def docking_options(self, item): return [np.array([0., 0., -1.35])]
            def advance(self, goal, **kwargs):
                self.position = np.asarray(goal)
                return dict(status='succeeded', reached=True)
        class Detector:
            def detect(self, *args, **kwargs): return [detection(), detection(3)]
        class Policy:
            program_control = True
            calls = 0
            def decide(self, instruction, obs, memory, selected, at_goal):
                self.calls += 1
                if self.calls == 1:
                    return dict(action='defer', candidate_id='v0001', reason='inconclusive')
                return dict(action='select', candidate_id='v0002', evidence_id=obs['id'])
        policy = Policy()
        result = VisualNavigator(IO(), Detector(), policy, self.tmp.name,
                                 max_observations=2, task=self.policy.task).run('chair')
        self.assertEqual(result['selected'], 'v0002')
        self.assertIn('v0001', policy.deferred_candidate_ids)
        self.assertEqual(policy.calls, 2)

        class RecoveredPolicy(Policy):
            def decide(self, instruction, obs, memory, selected, at_goal):
                self.calls += 1
                if self.calls == 1:
                    return dict(action='defer', candidate_id='v0001', reason='inconclusive')
                self.deferred_candidate_ids = []
                return dict(action='select', candidate_id='v0001', evidence_id=obs['id'])
        recovered = VisualNavigator(IO(), Detector(), RecoveredPolicy(), self.tmp.name,
                                    max_observations=2, task=self.policy.task).run('chair')
        self.assertEqual(recovered['selected'], 'v0001')
        self.assertTrue(any(e['type']=='candidate_reactivated' for e in recovered['events']))

    def test_same_frame_alternative_checked_before_uncertain_approach(self):
        obs = self.observe('o1', (0, 3))
        def answer(system, content):
            value = self.answer(system, content)
            if value['candidate_id'] == 'v0001':
                value.update(attributes_match='unknown', approach_recommended=True)
            return value
        self.policy._request.side_effect = answer
        result = self.policy.decide('', obs, self.memory)
        self.assertEqual(result['candidate_id'], 'v0002')

    def test_unidentified_dark_patch_does_not_authorize_direct_approach(self):
        obs = self.observe('o1')
        def answer(system, content):
            return dict(self.answer(system, content), category_match='unknown', attributes_match='unknown',
                        approach_recommended=True)
        self.policy._request.side_effect = answer
        self.assertEqual(self.policy.decide('', obs, self.memory)['action'], 'observe')


if __name__ == '__main__':
    unittest.main()
