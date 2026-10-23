"""Evidence provenance, revision and model boundary tests for visual relations."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PIL import Image
from visual_nav.relations import RelationMemory, pair_relations
from visual_nav.memory import ObjectMemory
from visual_nav.policy import VLMPolicy


def box(x=0, y=0):
    return dict(bbox_min=[x, y, 0], bbox_max=[x+.4, y+.4, .4], confidence=.8)


def obs(key, path='frame.jpg'):
    return dict(id=key, time=1., rgb_path=path, camera_to_world=np.eye(4), position=[0,0,0])


class RelationTests(unittest.TestCase):
    def test_only_joint_views_support_and_absence_does_not_erase(self):
        memory = RelationMemory()
        pair = {'a': box(), 'b': box(.8)}
        memory.update({'a': pair['a']}, obs('o1'))
        memory.update({'b': pair['b']}, obs('o2'))
        self.assertEqual(memory.summary(), [])
        memory.update(pair, obs('o3'))
        edge = memory.summary()[0]
        self.assertEqual(edge['status'], 'hypothesis')
        self.assertEqual(edge['evidence'][0]['rgb_path'], 'frame.jpg')
        memory.update({}, obs('o4'))
        self.assertEqual(memory.summary()[0], edge)
        memory.update(pair, obs('o5'))
        self.assertEqual(memory.summary()[0]['status'], 'supported')
        memory.update(pair, obs('o5'))
        self.assertEqual(memory.summary()[0]['supporting_observations'], 2)

    def test_joint_geometry_can_contradict_and_later_restore(self):
        memory = RelationMemory()
        memory.update({'a': box(), 'b': box(.8)}, obs('o1'))
        memory.update({'a': box(), 'b': box(4)}, obs('o2'))
        edge = memory.summary()[0]
        self.assertEqual(edge['status'], 'contradicted')
        self.assertFalse(edge['evidence'][-1]['supports'])
        memory.update({'a': box(), 'b': box(.8)}, obs('o3'))
        self.assertEqual(memory.summary()[0]['status'], 'hypothesis')

    def test_above_is_directional_and_duplicate_surface_is_not_a_relation(self):
        self.assertIn('a_above_b', pair_relations(box(y=.8), box()))
        self.assertNotIn('a_above_b', pair_relations(box(x=3, y=.8), box()))
        self.assertEqual(pair_relations(box(), box()), {})
        self.assertEqual(pair_relations(box(), box(4)), {})
        self.assertNotIn('on', pair_relations(box(y=.8), box()))

    def test_memory_and_policy_keep_provenance_without_sending_file_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, current = Path(tmp)/'old.jpg', Path(tmp)/'current.jpg'
            Image.new('RGB', (32, 32), 'white').save(old)
            Image.new('RGB', (32, 32), 'black').save(current)
            memory = ObjectMemory()
            detections = []
            for x, category in [(0, 'chair'), (.8, 'table')]:
                detections.append(dict(category=category, confidence=.8,
                    points=np.random.RandomState(1).uniform(0, .4, (100,3))+[x,0,0],
                    crop=str(old), mask='mask.png', attributes={}))
            memory.integrate(detections, obs('o1', str(old)))
            observation = obs('o2', str(current))
            memory.integrate([], observation)
            self.assertEqual(len(memory.summary()['relations']), 1)
            memory.relations.edges[('v0001', 'near', 'v0002')]['secret_gt_table'] = 'DO_NOT_SEND'
            policy = VLMPolicy.__new__(VLMPolicy)
            policy.task = dict(target_categories=['chair'], reference_categories=['table'], relation='near')
            policy._request = Mock(return_value=dict(action='select', candidate_id='v0001'))
            policy.decide('去桌旁的椅子', observation, memory)
            system, content = policy._request.call_args.args
            context = json.loads(content[0]['text'])
            self.assertEqual(context['relations'][0]['evidence_ids'], ['o1'])
            self.assertFalse(context['relations'][0]['current'])
            self.assertNotIn('DO_NOT_SEND', json.dumps(content))
            self.assertNotIn(tmp, json.dumps(context))
            self.assertIn('historical joint view o1', [x.get('text') for x in content])
            self.assertEqual(sum(x['type']=='image_url' for x in content), 4)


if __name__ == '__main__':
    unittest.main()
