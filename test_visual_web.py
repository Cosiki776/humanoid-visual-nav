"""Web boundary regressions: validate commands, isolate GT history, restrict assets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from visual_web import VisualApp

class VisualWebTests(unittest.TestCase):
    def test_assets_reject_traversal_and_non_image_files(self):
        app=VisualApp()
        with tempfile.TemporaryDirectory() as tmp, patch('visual_web.bootstrap.ROOT',Path(tmp)):
            root=Path(tmp)/'outputs'/'visual_run';root.mkdir(parents=True)
            (root/'obs_00001.jpg').write_bytes(b'image')
            (root/'observer_last.jpg').write_bytes(b'observer')
            self.assertEqual(app.asset('visual_run','observer_last.jpg').read_bytes(),b'observer')
            self.assertEqual(app.asset('visual_run','obs_00001.jpg').read_bytes(),b'image')
            for run,name in [('../','model_config.json'),('visual_run','result.json'),('visual_run','../obs_00001.jpg')]:
                with self.assertRaises(ValueError): app.asset(run,name)
            (root/'obs_00002.jpg').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError): app.asset('visual_run','obs_00002.jpg')

    def test_gt_results_never_appear_in_visual_history(self):
        app=VisualApp()
        with tempfile.TemporaryDirectory() as tmp, patch('visual_web.bootstrap.ROOT',Path(tmp)):
            for name,policy,mode in [('gt','ModelPlanner','gt'),('visual','VLMPolicy','active_verify')]:
                root=Path(tmp)/'outputs'/name;root.mkdir(parents=True)
                (root/'result.json').write_text(json.dumps(dict(policy=policy,mode=mode,status='succeeded',selected_points=[[1,2,3]])))
            self.assertEqual([r['id'] for r in app.runs()],['visual'])
            with self.assertRaises(ValueError): app.load('gt')
            snapshot=app.load('visual')
            self.assertTrue(snapshot['replay'])
            self.assertNotIn('selected_points',snapshot)
            self.assertNotIn('api_key',snapshot)
            app.busy=True
            with self.assertRaises(ValueError): app.load('visual')

    def test_command_validation_and_busy_guard_prevent_launch(self):
        app=VisualApp()
        with patch('visual_web.subprocess.Popen') as launch:
            for data in [dict(instruction=''),dict(instruction='chair',max_observations=0),dict(instruction='chair',max_observations=True)]:
                with self.assertRaises(ValueError): app.submit(data)
            app.busy=True
            with self.assertRaises(ValueError): app.submit(dict(instruction='chair'))
            launch.assert_not_called()

if __name__=='__main__': unittest.main()
