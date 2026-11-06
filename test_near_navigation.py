import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from PIL import Image
from test_visual_nav import observation, detection
from visual_nav.memory import ObjectMemory
from visual_nav.policy import VLMPolicy, DecisionValidationError
from visual_nav.loop import validate_target_change
from visual_nav.near_navigation import relation_evidence


class NearNavigationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name)/'image.jpg')
        Image.new('RGB', (100,100), 'white').save(self.path)
        self.memory = ObjectMemory()
        self.policy = VLMPolicy.__new__(VLMPolicy)
        self.policy.program_control = True
        self.policy.task = dict(target_categories=['chair'], reference_categories=['refrigerator'], attributes=['white'], relation='near')
        self.policy._request = Mock(side_effect=self.answer)

    def answer(self, system, content):
        c = json.loads(content[0]['text'])
        return dict(candidate_id=c['candidate_id'], reference_id=c['reference_id'], observation_id=c['observation_id'],
                    category_match='yes', attributes_match='yes', reference_match='yes', relation_match='yes', reason='Visual evidence')

    def observe(self, key, x=0, reference=True, time=1):
        obs = observation(key)
        obs.update(rgb_path=self.path, time=time, position=[x,0,0])
        obs['camera_to_world'][0,3] = x
        detections = [detection()]
        if reference:
            d = detection(.8)
            d['category'] = 'refrigerator'
            detections.append(d)
        for d in detections:
            d.update(crop=self.path, bbox=[10,10,50,50])
        self.memory.integrate(detections, obs)
        return obs

    def test_cache_and_historical_relation_confirmation(self):
        obs = self.observe('o1')
        self.assertEqual(self.policy.decide('',obs,self.memory)['action'],'select')
        obs = self.observe('o2',x=.6,time=2)
        self.assertEqual(self.policy.decide('',obs,self.memory,'v0001')['action'],'select')
        self.assertEqual(self.policy._request.call_count,1)
        obs = self.observe('o3',reference=False,time=3)
        decision = self.policy.decide('',obs,self.memory,'v0001',True)
        self.assertEqual(decision['action'],'confirm')
        validate_target_change(decision,self.memory,self.policy.task,'v0001',True,True)
        self.assertEqual(self.policy._request.call_count,2)
        self.assertEqual(self.policy.last_input['image_count'],4)

    def test_single_view_missing_reference_requires_observation(self):
        self.policy.decide('',self.observe('o1'),self.memory)
        obs = self.observe('o2',reference=False,time=2)
        self.assertEqual(self.policy.decide('',obs,self.memory,'v0001',True)['action'],'observe')
        self.assertEqual(self.policy._request.call_count,1)

    def test_same_pose_and_stale_views_not_sufficient(self):
        self.observe('o1')
        self.observe('o2',time=2)
        obs = self.observe('o3',reference=False,time=3)
        self.assertIsNone(relation_evidence(self.memory,'v0001','v0002',obs,True))
        self.observe('o4',x=.6,time=4)
        obs = self.observe('o5',reference=False,time=200)
        self.assertIsNone(relation_evidence(self.memory,'v0001','v0002',obs,True))

    def test_wrong_reference_is_not_accepted(self):
        obs = self.observe('o1')
        self.policy._request.side_effect=lambda s,c: dict(self.answer(s,c),reference_match='no')
        self.assertEqual(self.policy.decide('',obs,self.memory)['action'],'explore')

    def test_invented_id_is_rejected(self):
        obs = self.observe('o1')
        self.policy._request.side_effect=lambda s,c: dict(self.answer(s,c),reference_id='fake')
        with self.assertRaises(DecisionValidationError): self.policy.decide('',obs,self.memory)

    def test_contradicted_geometry_blocks_confirm(self):
        self.policy.decide('',self.observe('o1'),self.memory)
        next(iter(self.memory.relations.edges.values()))['status']='contradicted'
        obs = self.observe('o2',reference=False,time=2)
        self.assertEqual(self.policy.decide('',obs,self.memory,'v0001',True)['action'],'observe')
