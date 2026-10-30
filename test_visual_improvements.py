"""Regressions for identity, evidence budgets, viewpoints and inference concurrency."""
import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from test_visual_nav import observation, detection
from visual_nav.memory import ObjectMemory
from visual_nav.geometry import visible_points
from visual_nav.views import ViewMemory
from visual_nav.inspection import rank_viewpoints, reference_for
from visual_nav.async_runtime import AsyncInference
from visual_nav.loop import VisualNavigator, validate_target_change
from visual_nav.policy import VLMPolicy, DecisionValidationError


class EvidenceTests(unittest.TestCase):
    def test_global_assignment_does_not_let_confidence_steal_identity(self):
        memory = ObjectMemory()
        memory.integrate([detection(0), detection(.4)], observation())
        ambiguous, clear = detection(.16), detection(.01)
        ambiguous['confidence'], clear['confidence'] = .99, .5
        seen = memory.integrate([ambiguous, clear], observation('o2'))
        self.assertEqual(seen, ['v0002', 'v0001'])
        self.assertEqual(len(memory.instances), 2)

    def test_occluded_neighbors_do_not_trigger_repeated_panorama(self):
        from visual_nav.search import CoverageSearch
        search = CoverageSearch([[0,0,0], [0,0,-1]], lambda a,b: True)
        obs = observation()
        obs.update(depth=np.full((9,9), .3), intrinsics=np.array([[4,0,4],[0,4,4],[0,0,1]]))
        for direction in range(4):
            search.record_view(np.zeros(3),direction*np.pi/2,'o'+str(direction),obs)
        self.assertNotIn(1,search.covered)
        self.assertIsNone(search.next_view(np.zeros(3)))
        self.assertEqual(search.next_goal(np.zeros(3)).tolist(),[0,0,-1])

    def test_depth_blocks_hidden_search_samples(self):
        obs = observation()
        obs.update(depth=np.ones((9,9)), intrinsics=np.array([[4,0,4],[0,4,4],[0,0,1]]))
        visible = visible_points([[0,0,-.5], [0,0,-3], [3,0,-.5], [0,0,1]], obs)
        self.assertEqual(visible.tolist(), [True, False, False, False])

    def test_scene_views_deduplicate_and_retrieve_joint_evidence(self):
        views = ViewMemory(capacity=2)
        views.add(observation('o1'), ['a'])
        views.add(observation('o2'), ['a', 'b'])
        self.assertEqual(len(views.views), 1)
        moved = observation('o3'); moved['camera_to_world'][0,3] = 2
        views.add(moved, ['a'])
        self.assertEqual(views.retrieve(['a','b'])[0]['id'], 'o2')
        views.add(observation('o4'), ['c'])
        self.assertEqual(len(views.views), 2)
        self.assertEqual(views.retrieve(['a','b'], exclude=['o2'])[0]['id'], 'o3')

    def test_deferred_candidate_cannot_restart_inspection(self):
        with self.assertRaises(DecisionValidationError):
            validate_target_change(dict(action='select', candidate_id='v1'), ObjectMemory(),
                                   {'deferred_candidate_ids':['v1']}, None, False)

    def test_adaptive_inputs_and_explicit_expansion_keep_legacy_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'image.jpg'; Image.new('RGB',(16,16)).save(path)
            memory = ObjectMemory(); obs = observation(); obs['rgb_path'] = str(path)
            ds = [detection(i) for i in range(12)]
            for d in ds: d['crop'] = str(path)
            memory.integrate(ds, obs)
            policy = object.__new__(VLMPolicy)
            policy.task = dict(target_categories=['chair'], reference_categories=[])
            inputs = []
            def request(system, content):
                inputs.append(content)
                return dict(action='select', candidate_id='v0001')
            policy._request = request
            with patch.dict('os.environ', {'NAV_IMAGE_POLICY':'adaptive'}):
                policy.decide('chair',obs,memory)
                self.assertEqual(policy.last_input['image_count'],6)
                policy.decide('chair',obs,memory,'v0001',False)
                self.assertEqual(policy.last_input['image_count'],4)
                policy.expand_evidence=True
                policy.decide('chair',obs,memory)
                self.assertEqual(policy.last_input['image_count'],11)
            with patch.dict('os.environ', {'NAV_IMAGE_POLICY':'legacy'}):
                policy.decide('chair',obs,memory)
                self.assertEqual(policy.last_input['image_count'],11)
            self.assertNotIn(tmp, inputs[-1][0]['text'])

    def test_joint_viewpoint_rejects_occlusion_and_visited_positions(self):
        target = dict(position=[0,.8,-2]); ref = dict(position=[.6,.8,-2])
        points = [[0,0,0],[1,0,0],[2,0,0]]
        route = lambda p: [[-1,0,0],p]
        visible = lambda eye, item: eye[0] != 1
        ranked = rank_viewpoints(points, [-1,0,0], target, ref, route, visible, visited=[[0,0,0]])
        self.assertEqual(len(ranked),1)
        self.assertEqual(ranked[0].tolist(),[2,0,0])


class ConcurrencyTests(unittest.TestCase):
    def test_async_geometry_fallback_continues_without_semantic_pause(self):
        from visual_nav.habitat_io import HabitatIO
        class Executor:
            position=np.zeros(3)
            trace=[[0,0,0]]
            def execute_segment(self,target,**kwargs):
                self.position=np.asarray(target).copy(); self.trace.append(self.position.tolist())
                return dict(status='succeeded')
            def stop(self): pass
        adapter=object.__new__(HabitatIO)
        adapter._executor=Executor(); adapter._roadmap=None
        adapter._world=SimpleNamespace(direct=lambda a,b: not np.allclose(b,[1,0,0]),pf=None,island=0)
        adapter.visited=[]; adapter.should_stop=lambda:False
        goal=np.array([2,0,-1])
        adapter.route=lambda endpoint:[np.zeros(3),np.array([1,0,0]),goal]
        roadmap=SimpleNamespace(path=lambda a,b:[a,np.array([0,0,-1]),goal])
        with patch('visual_nav.roadmap.GeometricRoadmap',return_value=roadmap):
            event=adapter.advance(goal,budget=10,on_step=lambda:False)
        self.assertTrue(event['reached'])
        self.assertEqual(len(adapter._executor.trace),3)

    def test_short_remaining_route_does_not_launch_redundant_detection(self):
        from visual_nav.async_runtime import worthwhile_sample
        self.assertFalse(worthwhile_sample(1.2,4.0))
        self.assertTrue(worthwhile_sample(5.0,4.0))

    def test_single_flight_copies_observation_and_memory(self):
        entered, release = threading.Event(), threading.Event()
        owner = threading.get_ident()
        class Detector:
            def detect(self, obs, *args, **kwargs):
                self.thread = threading.get_ident(); entered.set(); release.wait(2)
                return [dict(pixel=int(obs['rgb'][0,0,0]))]
        class Policy:
            def decide(self, instruction, obs, memory, selected, at_goal):
                release.wait(2)
                return dict(size=len(memory.instances), at_goal=at_goal)
        detector=Detector(); runtime=AsyncInference(detector,Policy(),'.',('chair',))
        try:
            obs=observation(); obs['rgb']=np.zeros((2,2,3),dtype=np.uint8)
            self.assertTrue(runtime.submit_observation(obs)); self.assertTrue(entered.wait(1))
            obs['rgb'][:]=255
            self.assertFalse(runtime.submit_observation(obs))
            memory=ObjectMemory(); runtime.submit_decision('chair',obs,memory,None)
            memory.integrate([detection()],observation())
            release.set()
            result=runtime.take_observation(wait=True)
            self.assertEqual(result[1][0]['pixel'],0)
            self.assertNotEqual(detector.thread,owner)
            self.assertEqual(runtime.take_decision(wait=True),dict(size=0,at_goal=False))
        finally:
            release.set(); runtime.close()
        self.assertFalse(runtime.submit_observation(obs))

    def test_worker_exception_is_propagated(self):
        class Detector:
            def detect(self,*args,**kwargs): raise RuntimeError('perception failed')
        runtime=AsyncInference(Detector(),None,'.',())
        try:
            runtime.submit_observation(observation())
            with self.assertRaisesRegex(RuntimeError,'perception failed'):
                runtime.take_observation(wait=True)
        finally: runtime.close()

    def test_async_navigation_uses_fresh_stationary_confirmation(self):
        owner=threading.get_ident()
        class IO:
            start=[0,0,0]; position=np.zeros(3); counter=0; yaw=0.
            def observe(self,**kwargs):
                assert threading.get_ident()==owner
                self.counter+=1; obs=observation('o'+str(self.counter))
                obs['position']=self.position.tolist()
                return obs
            def docking_options(self,item): return [np.array([0,0,-3.35])]
            def advance(self,endpoint,budget=3.,on_step=None):
                for i in range(5):
                    if on_step and on_step():
                        return dict(status='interrupted',reached=False)
                    self.position=self.position+(np.asarray(endpoint)-self.position)/(5-i)
                    time.sleep(.01)
                return dict(status='succeeded',reached=True)
            def trajectory(self): return [self.start,self.position.tolist()]
        class Detector:
            def detect(self,*args,**kwargs):
                d=detection(); d['points']+=np.array([0,0,-2]); return [d]
        class Policy:
            calls=0
            def decide(self,instruction,obs,memory,selected,at_goal):
                self.calls+=1
                return dict(action='confirm' if at_goal else 'select', candidate_id='v0001',evidence_id=obs['id'])
        ticks=iter(range(10000))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), patch(
                'visual_nav.loop.time.monotonic',side_effect=lambda:float(next(ticks))):
            nav=VisualNavigator(IO(),Detector(),Policy(),tmp,execution_mode='async')
            result=nav.run('chair')
        self.assertTrue(result['verified'])
        moving=[e for e in result['events'] if e.get('asynchronous')]
        self.assertGreater(len(moving),0)
        self.assertNotIn(result['events'][-1]['evidence_id'],[e['id'] for e in moving])
        self.assertTrue(nav.runtime.closed)


if __name__=='__main__': unittest.main()
