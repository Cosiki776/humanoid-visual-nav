"""Fixed conversational reference, evidence-gated switching and bounded recovery."""
import tempfile
import unittest
import numpy as np
from tests.test_visual_nav import detection, observation
from visual_nav.memory import ObjectMemory
from visual_nav.task import simple_navigation_task
from visual_nav.task_context import bind_task, excluded_ids, goal_snapshot, requests_another
from visual_nav.policy import DecisionValidationError
from visual_nav.loop import VisualNavigator, validate_target_change


class ContextTests(unittest.TestCase):
    def test_relation_confirmation_needs_current_reference_and_geometry(self):
        memory = ObjectMemory()
        reference = detection(.8)
        reference['category'] = 'refrigerator'
        memory.integrate([detection(), reference], observation())
        task = dict(relation='near', target_categories=['chair'], reference_categories=['refrigerator'])
        decision = dict(action='confirm', candidate_id='v0001')
        with self.assertRaises(DecisionValidationError):
            validate_target_change(decision, memory, task, 'v0001', True)
        decision.update(reference_id='v0002', reference_verified=True, reference_reason='Visible refrigerator door and handle')
        validate_target_change(decision, memory, task, 'v0001', True)
        memory.integrate([detection()], observation('o2'))
        with self.assertRaises(DecisionValidationError):
            validate_target_change(decision, memory, task, 'v0001', True)

    def test_ambiguous_candidate_cannot_interrupt_search_forever(self):
        from visual_nav.search import CoverageSearch
        class IO:
            start = [0,0,0]
            def __init__(self):
                self.position = np.zeros(3)
                self.counter = 0
                self.search = CoverageSearch([[0,0,-1.35], [5,0,0]], lambda a,b: True)
            def observe(self, yaw=0, look_at=None):
                self.counter += 1
                obs = observation('o'+str(self.counter))
                obs['position'] = self.position.tolist()
                if look_at is None: self.search.record_view(self.position, yaw, obs['id'])
                return obs
            def search_yaw(self): return self.search.next_view(self.position)
            def search_summary(self): return self.search.summary()
            def explore_goal(self): return self.search.next_goal(self.position)
            def mark_search_failure(self, goal): self.search.failed(goal)
            def docking_options(self, item): return [np.array([0,0,-1.35])]
            def advance(self, endpoint, **kwargs):
                self.position = np.asarray(endpoint).copy()
                return dict(status='succeeded', reached=True)
            def trajectory(self): return [self.start, self.position.tolist()]
        class Detector:
            def detect(self, *args, **kwargs): return [detection()]
        class Policy:
            calls = 0
            def decide(self, instruction, obs, *args):
                self.calls += 1
                return dict(action='select', candidate_id='v0001', evidence_id=obs['id'])
        with tempfile.TemporaryDirectory() as tmp:
            io = IO()
            result = VisualNavigator(io, Detector(), Policy(), tmp, max_observations=24).run('chair')
            self.assertTrue(any(e['type']=='search_resume' for e in result['events']))
            self.assertTrue(any(e.get('purpose')=='search' and e['destination']==[5.,0.,0.] for e in result['events']))
            self.assertEqual(result['status'], 'search_complete_no_match')
            self.assertFalse(result['verified'])

    def test_another_keeps_original_anchor_and_excludes_duplicate_ids(self):
        memory = ObjectMemory()
        memory.integrate([detection(0), detection(3)], observation())
        previous = goal_snapshot(memory.instances['v0001'])
        task = bind_task(simple_navigation_task('去另外一个白色椅子旁边'), '去另外一个白色椅子旁边', previous)
        previous['id'] = 'changed'
        self.assertEqual(task['excluded_goal']['id'], 'v0001')
        memory.instances['duplicate'] = dict(memory.instances['v0001'], id='duplicate')
        self.assertEqual(excluded_ids(task, memory.instances), {'v0001', 'duplicate'})
        # Arriving at v0002 never changes the old anchor or excludes the new target.
        validate_target_change(dict(action='confirm', candidate_id='v0002'), memory, task, 'v0002', True)
        with self.assertRaises(DecisionValidationError):
            validate_target_change(dict(action='select', candidate_id='duplicate'), memory, task, None, False)
        with self.assertRaises(DecisionValidationError):
            validate_target_change(dict(action='explore'), memory, task, 'v0002', True)
        with self.assertRaises(ValueError):
            bind_task(task, '去另一个白色椅子', None)

    def test_full_instruction_and_switch_need_new_evidence(self):
        self.assertFalse(requests_another('去桌子的另外一边'))
        memory = ObjectMemory()
        memory.integrate([detection(0), detection(3)], observation())
        task = bind_task(simple_navigation_task('去白色椅子旁边'),
                         '你现在已经到达这个白色椅子了，现在去另一个白色椅子旁边',
                         goal_snapshot(memory.instances['v0001']))
        self.assertTrue(task['another'])
        decision = dict(action='select', candidate_id='v0002', switch_reason='新的完整椅子证据')
        with self.assertRaises(DecisionValidationError):
            validate_target_change(decision, memory, None, 'v0001', False)

        memory.integrate([detection(0), detection(3)], observation('o2'))
        validate_target_change(decision, memory, None, 'v0001', False)
        memory.instances['v0002']['confidence'] = .3
        with self.assertRaises(DecisionValidationError):
            validate_target_change(decision, memory, None, 'v0001', False)

    def test_touching_partial_surfaces_cannot_masquerade_as_another_chair(self):
        anchor = dict(id='old', category='chair', position=[0,.5,0],
                      bbox_min=[-.2,.4,-.2], bbox_max=[.2,1.2,.1], last_seen='o1')
        partial = dict(id='new', category='chair', position=[0,1,.25],
                       bbox_min=[-.2,.6,.08], bbox_max=[.2,1.4,.4])
        other = dict(partial, id='other', position=[2,1,.25], bbox_min=[1.8,.6,.08], bbox_max=[2.2,1.4,.4])
        self.assertEqual(excluded_ids({'excluded_goal': anchor}, {'new': partial, 'other': other}), {'old', 'new'})

    def test_bad_confirmation_reobserves_and_is_bounded(self):
        class IO:
            start = [0,0,0]
            position = np.zeros(3)
            counter = 0
            def observe(self, **kwargs):
                self.counter += 1
                return observation('o'+str(self.counter))
            def docking_options(self, instance):
                return [np.array([0,0,-1.35])]
            def advance(self, endpoint, **kwargs):
                self.position = np.asarray(endpoint)
                return dict(status='succeeded', reached=True)
            def trajectory(self):
                return [self.start, self.position.tolist()]
        class Policy:
            calls = 0
            def decide(self, instruction, obs, memory, selected, at_goal):
                return dict(action='confirm' if at_goal else 'select', candidate_id='v0001', evidence_id=obs['id'])
        class Detector:
            def __init__(self, permanent=False): self.permanent=permanent
            def detect(self, obs, output, categories):
                return [] if obs['id']=='o2' or (self.permanent and obs['id']!='o1') else [detection()]
        with tempfile.TemporaryDirectory() as tmp:
            recovered = VisualNavigator(IO(), Detector(), Policy(), tmp).run('chair')
            self.assertTrue(recovered['verified'])
            self.assertEqual(recovered['observations'], 3)
            self.assertEqual(sum(e['type']=='decision_retry' for e in recovered['events']), 1)
            bounded = VisualNavigator(IO(), Detector(True), Policy(), tmp).run('chair')
            self.assertEqual(bounded['status'], 'decision_unresolved')
            self.assertFalse(bounded['verified'])
            self.assertEqual(bounded['observations'], 4)
            self.assertEqual(sum(e['type']=='execution' for e in bounded['events']), 1)


if __name__ == '__main__': unittest.main()
