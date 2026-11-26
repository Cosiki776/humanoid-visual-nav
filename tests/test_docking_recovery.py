"""Unavailable final docking must allow bounded approach, never false success."""
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from tests.test_visual_nav import observation, detection
from visual_nav.loop import VisualNavigator
from visual_nav.habitat_io import HabitatIO


class RecoveryTests(unittest.TestCase):
    def run_navigation(self, blocked=False):
        class IO:
            start = [0, 0, 0]
            def __init__(self):
                self.position = np.zeros(3)
                self.count = 0
                self.moves = []
            def observe(self, **kwargs):
                self.count += 1
                result = observation('o'+str(self.count))
                result['position'] = self.position.tolist()
                return result
            def docking_options(self, item):
                return [] if blocked or self.count == 1 else [np.array([0., 0., -1.35])]
            def approach_options(self, item, visited):
                return [np.array([0., 0., -1.35 if blocked else -.8])] if self.count == 1 else []
            def advance(self, goal, **kwargs):
                self.position = np.asarray(goal).copy()
                self.moves.append(self.position.tolist())
                return dict(status='succeeded', reached=True)
            def trajectory(self): return [self.start]+self.moves
        class Detector:
            def detect(self, *args, **kwargs): return [detection()]
        class Policy:
            calls = 0
            def decide(self, instruction, obs, memory, selected, at_goal):
                return dict(action='confirm' if at_goal else 'select', candidate_id='v0001', evidence_id=obs['id'])
        with tempfile.TemporaryDirectory() as tmp:
            return VisualNavigator(IO(), Detector(), Policy(), tmp, max_observations=8).run('chair')

    def test_missing_dock_approaches_then_reobserves_before_final_arrival(self):
        result = self.run_navigation()
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(result['observations'], 3)
        self.assertTrue(any(e['type']=='approach_plan' for e in result['events']))
        self.assertEqual(result['events'][-1]['evidence_id'], 'o3')

    def test_approach_even_inside_arrival_band_cannot_confirm(self):
        result = self.run_navigation(blocked=True)
        self.assertEqual(result['status'], 'target_approach_unresolved')
        self.assertFalse(result['verified'])
        self.assertEqual(result['selected'], 'v0001')
        self.assertEqual(result['observations'], 2)
        self.assertFalse(any(e.get('action')=='confirm' for e in result['events']))

    def test_diagnostics_explain_navmesh_snap_rejections(self):
        io = object.__new__(HabitatIO)
        io._executor = SimpleNamespace(position=np.zeros(3))
        io._world = SimpleNamespace(island=0, pf=SimpleNamespace(snap_point=lambda p,i: np.asarray(p)+[0,.2,0]))
        io.route = lambda point: self.fail('Rejected samples must not invoke route planning')
        item = dict(bbox_min=[-.2,0,-2.2], bbox_max=[.2,1,-1.8], points=detection()['points'])
        self.assertEqual(io.docking_options(item), [])
        self.assertGreater(io.last_docking_diagnostics['sampled'], 0)
        self.assertEqual(io.last_docking_diagnostics['sampled'], io.last_docking_diagnostics['height_rejected'])

    def test_lateral_snap_can_produce_a_valid_final_dock(self):
        io = object.__new__(HabitatIO)
        io._executor = SimpleNamespace(position=np.zeros(3))
        point = np.array([0., 0., -1.35])
        io._world = SimpleNamespace(island=0, pf=SimpleNamespace(snap_point=lambda p,i: point.copy()))
        io.route = lambda p: [np.zeros(3), p]
        item = dict(bbox_min=[-.15,.65,-2.15], bbox_max=[.15,.95,-1.85], points=detection()['points'])
        options = io.docking_options(item)
        self.assertEqual(len(options), 1)
        self.assertGreater(io.last_docking_diagnostics['large_snap'], 0)


if __name__ == '__main__':
    unittest.main()
