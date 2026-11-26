import unittest
from types import SimpleNamespace
import numpy as np
from visual_nav.directional import parse_directional, direction_vector, move_straight


class DirectionalTests(unittest.TestCase):
    def test_parser(self):
        for text,direction,distance in [('向前走2米','forward',2),('后退0.5米','backward',.5),('往左走两米','left',2),('向右移动半米','right',.5),('向前走一点五米','forward',1.5)]:
            self.assertEqual(parse_directional(text),dict(kind='directional_move',direction=direction,distance_m=distance))
        for text in ['去白色椅子旁边','去左边的柜子','向前走2米再向左走1米']:
            self.assertIsNone(parse_directional(text))
        for text in ['向前走-1米','向前走0米','向前走100米']:
            with self.assertRaises(ValueError):parse_directional(text)

    def test_body_relative_axes(self):
        np.testing.assert_allclose(direction_vector([0,0,-1],'left'),[-1,0,0])
        np.testing.assert_allclose(direction_vector([1,0,0],'left'),[0,0,-1])

    def run_motion(self,limit=10,cancel=False):
        class Executor:
            position=np.array([0.,0.,0.])
            controller=SimpleNamespace(prev_orientation=np.array([0.,0.,-1.]))
            def turn_to(self,direction,should_stop):
                self.controller.prev_orientation=direction
                return dict(status='succeeded')
            def execute_segment(self,endpoint,**kwargs):
                self.position=np.asarray(endpoint)
                return dict(status='succeeded')
            def stop(self):pass
        executor=Executor()
        world=SimpleNamespace(direct=lambda a,b: abs(b[2])<=limit+1e-8)
        return move_straight(executor,world,'forward',2,lambda:cancel)

    def test_full_distance(self):
        result=self.run_motion()
        self.assertEqual(result['status'],'succeeded')
        self.assertAlmostEqual(result['travelled'],2)

    def test_wall_stops_at_prefix_without_detour(self):
        result=self.run_motion(.83)
        self.assertEqual(result['status'],'blocked')
        self.assertAlmostEqual(result['travelled'],.8)
        np.testing.assert_allclose(result['position'],[0,0,-.8])

    def test_blocked_immediately_and_cancelled(self):
        self.assertEqual(self.run_motion(0)['travelled'],0)
        self.assertEqual(self.run_motion(cancel=True)['status'],'cancelled')
