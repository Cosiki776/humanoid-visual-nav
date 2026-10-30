"""Observation-group boundaries, source provenance, and compact VLM packets."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PIL import Image
from visual_nav.views import ViewMemory
from visual_nav.memory import ObjectMemory
from visual_nav.policy import VLMPolicy
from test_visual_nav import observation


def frame(index, yaw=0., x=0., kind='scan'):
    obs = observation('o'+str(index))
    obs['position'] = [x, 0, 0]
    obs['camera_to_world'] = np.array([[math.cos(yaw),0,-math.sin(yaw),x],
                                     [0,1,0,1.65], [math.sin(yaw),0,math.cos(yaw),0], [0,0,0,1.]])
    obs['time'] = index
    obs['capture_kind'] = kind
    return obs


class ObservationMemoryTests(unittest.TestCase):
    def test_stationary_sweep_is_one_group_with_four_directions(self):
        m = ViewMemory()
        for i in range(4): m.add(frame(i, i*math.pi/2), ['chair'] if i==0 else [])
        groups = m.observation_groups()
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]['complete'])
        self.assertEqual([f['direction'] for f in groups[0]['frames']], ['front','right','back','left'])
        self.assertNotIn('image', groups[0]['frames'][0])  # Only references, no duplicate image paths.
        self.assertEqual(m.views[0]['camera_to_world'][1][3], 1.65)

    def test_motion_and_keyframes_break_groups(self):
        m = ViewMemory()
        m.add(frame(0), [])
        m.add(frame(1, math.pi/2, x=1), [])
        m.add(frame(2, math.pi, x=1, kind='keyframe'), [])
        m.add(frame(3, -math.pi/2, x=1), [])
        self.assertEqual(len(m.groups), 4)
        self.assertFalse(any(g['complete'] for g in m.observation_groups()))

    def test_repeated_direction_is_not_four_directions(self):
        m = ViewMemory()
        for i in range(4): m.add(frame(i, .01*i), ['a'] if i==0 else [])
        self.assertEqual(len(m.views), 1)
        g = m.observation_groups()[0]
        self.assertFalse(g['complete'])
        self.assertEqual(g['frames'][0]['visible_objects'], [])

    def test_capacity_prunes_references_and_does_not_claim_full_coverage(self):
        m = ViewMemory(capacity=3)
        for i in range(4): m.add(frame(i, i*math.pi/2), [])
        self.assertFalse(m.observation_groups()[0]['complete'])
        self.assertEqual(set(m.groups[0]['frame_ids']), {v['id'] for v in m.views})

    def test_context_is_bounded_and_not_absence_or_invented_rooms(self):
        m = ViewMemory()
        for i in range(16): m.add(frame(i, i*math.pi/2, x=i//4), [])
        context = m.context([], exclude={'o15'})
        self.assertLessEqual(len(context['history']), 4)
        self.assertEqual(len(context['frames']), 3)
        self.assertEqual(context['evidence'], [])
        self.assertIn('not proof', context['meaning'])
        self.assertNotIn('room', json.dumps(context))

    def test_vlm_gets_current_plus_three_directions_without_duplicate_history_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = ObjectMemory()
            for i in range(4):
                obs = frame(i, i*math.pi/2)
                path = Path(tmp)/(obs['id']+'.jpg')
                Image.new('RGB',(16,16)).save(path)
                obs['rgb_path'] = str(path)
                m.integrate([], obs)
            policy = VLMPolicy.__new__(VLMPolicy)
            policy.task = {'target_categories':['chair']}
            policy._request = Mock(return_value={'action':'explore'})
            policy.decide('找椅子', obs, m)
            _, content = policy._request.call_args.args
            packet = json.loads(content[0]['text'])
            self.assertEqual(sum(c['type']=='image_url' for c in content), 4)
            self.assertTrue(packet['observation_memory']['selected_group']['complete'])
            self.assertNotIn(tmp, content[0]['text'])
            self.assertNotIn('history', m.__dict__)

    def test_rejections_are_a_view_and_questions_have_one_owner(self):
        m = ObjectMemory()
        m.ledger.begin({'target_categories':['chair'], 'attributes':['white']})
        m.ledger.record('a','o1','brown','attribute_mismatch')
        m.rejections.clear()  # Modifying compatibility output cannot erase the ledger.
        self.assertEqual(len(m.ledger.claims), 1)
        self.assertIs(m.questions, m.ledger.questions)
        self.assertNotIn('rejections', m.__dict__)
        m.ledger.begin({'target_categories':['chair']})
        self.assertEqual(len(m.ledger.claims), 1)


if __name__ == '__main__': unittest.main()
