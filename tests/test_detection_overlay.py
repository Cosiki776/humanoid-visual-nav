"""Viewer boxes must belong to the exact observation, including empty frames."""
import unittest
from types import SimpleNamespace

from visual_nav.loop import VisualNavigator


class DetectionOverlayTests(unittest.TestCase):
    def test_only_matching_frame_evidence_is_published(self):
        nav = VisualNavigator.__new__(VisualNavigator)
        nav.memory = SimpleNamespace(instances={'v0001': dict(category='chair', evidence=[
            dict(observation_id='old', bbox=[1, 2, 3, 4], confidence=.9),
            dict(observation_id='new', bbox=[10, 20, 30, 40], confidence=.8)])})
        boxes = nav.detection_overlay(dict(id='new', rgb_path='new.jpg'), ['v0001'])
        self.assertEqual(boxes, [dict(id='v0001', category='chair',
                                     bbox=[10, 20, 30, 40], confidence=.8)])
        self.assertEqual(nav.current_observation['detections'], boxes)
        self.assertEqual(nav.detection_overlay(dict(id='empty', rgb_path='empty.jpg'), []), [])
        self.assertEqual(nav.current_observation['detections'], [])


if __name__ == '__main__':
    unittest.main()
