"""Meaningful RGB-D, evidence, permission, and independent-scoring regressions."""
import ast
import json
import tempfile
import unittest
import io
from unittest.mock import patch
from pathlib import Path
import numpy as np
from visual_nav.geometry import unproject, path_length
from visual_nav.memory import ObjectMemory
from visual_nav.policy import validate_decision, VLMPolicy
from visual_nav.loop import VisualNavigator
from visual_nav.roadmap import GeometricRoadmap
from visual_nav.search import CoverageSearch
from evaluate_visual import associate, score_episode, aggregate


def observation(key='o1'):
    return dict(id=key, time=1., camera_to_world=np.eye(4), rgb_path='image.jpg', position=[0,0,0])


def detection(offset=0, color='red'):
    # Deterministic cloud; includes finite extent in all axes.
    cloud = np.random.RandomState(4).uniform(-.15, .15, (100, 3)) + [offset, .8, -2]
    return dict(points=cloud, category='chair', confidence=.8, crop='crop.jpg', mask='mask.png',
                attributes={'color_scores': {color: 1.0}})


class GeometryTests(unittest.TestCase):
    def test_geometry_only_graph_routes_around_obstacle(self):
        def free(point):
            return not (-.45 < point[0] < .45 and -1.1 < point[2] < 1.1)
        def direct(a, b):
            return all(free(p) for p in np.linspace(a,b,80))
        class PF:
            def get_bounds(self):
                return [-2,0,-2], [2,0,2]
            def snap_point(self, point, island):
                return point if free(point) else [float('nan')]*3
        graph = GeometricRoadmap(PF(), 0, direct, 0, spacing=.4)
        route = graph.path([-1.6,0,0], [1.6,0,0])
        self.assertIsNotNone(route)
        self.assertGreater(path_length(route), 3.2)
        self.assertTrue(all(direct(a,b) for a,b in zip(route,route[1:])))

    def test_depth_is_optical_z_and_axes_are_habitat(self):
        depth = np.full((6,6), 2.)
        k = np.array([[2,0,2.5],[0,2,2.5],[0,0,1]])
        pose = np.eye(4)
        pose[:3, 3] = [1,3,4]
        points = unproject(depth, np.ones_like(depth, bool), k, pose)
        np.testing.assert_allclose(points[0], [-1.5,5.5,2])
        np.testing.assert_allclose(np.median(points, axis=0), [1,3,2])

    def test_invalid_depth_and_shape(self):
        depth = np.zeros((8,8))
        depth[:2] = np.nan
        self.assertIsNone(unproject(depth, np.ones_like(depth, bool), np.eye(3), np.eye(4)))
        with self.assertRaises(ValueError):
            unproject(depth, np.ones((1,1)), np.eye(3), np.eye(4))


class SearchTests(unittest.TestCase):
    def test_search_goal_is_stable_and_repeated_stalls_are_bounded(self):
        search = CoverageSearch([[0,0,0],[3,0,0]], lambda a,b: False)
        first = search.next_goal(np.zeros(3))
        np.testing.assert_allclose(search.next_goal(np.array([3,0,0])), first)
        for sweep in range(3):
            for direction in range(4):
                search.record_view([1.5,0,0], direction*np.pi/2, 'o%d_%d' % (sweep,direction))
        self.assertEqual(search.summary()['blocked_samples'], 1)
        np.testing.assert_allclose(search.next_goal(np.zeros(3)), [3,0,0])

    def test_single_frame_never_completes_a_search(self):
        search = CoverageSearch([[0,0,0],[.6,0,0]], lambda a,b: True)
        search.record_view([0,0,0], 0, 'o1')
        self.assertEqual(search.summary()['covered_samples'], 0)
        for i in range(1,4):
            search.record_view([0,0,0], i*np.pi/2, 'o'+str(i+1))
        self.assertTrue(search.summary()['complete'])
        self.assertIsNone(search.next_goal(np.zeros(3)))

    def test_wall_and_failed_region_remain_unresolved(self):
        search = CoverageSearch([[-.3,0,0],[.3,0,0]], lambda a,b: a[0]*b[0] > 0)
        for i in range(4):
            search.record_view([-.3,0,0], i*np.pi/2, 'o'+str(i))
        self.assertEqual(search.summary()['covered_samples'], 1)
        goal = search.next_goal(np.zeros(3))
        search.failed(goal)
        self.assertIsNotNone(search.next_goal(np.zeros(3)))
        search.failed(goal)
        self.assertIsNone(search.next_goal(np.zeros(3)))
        self.assertTrue(search.summary()['exhausted'])
        self.assertFalse(search.summary()['complete'])

    def test_navigation_stops_after_completed_sweep_without_target(self):
        class IO:
            start = [0,0,0]
            position = np.zeros(3)
            counter = 0
            search = CoverageSearch([[0,0,0]], lambda a,b: True)
            def observe(self, yaw=0, **kwargs):
                self.counter += 1
                obs = observation('o'+str(self.counter))
                self.search.record_view(self.position, yaw, obs['id'])
                return obs
            def explore_goal(self):
                return self.search.next_goal(self.position)
            def search_summary(self):
                return self.search.summary()
            def trajectory(self):
                return [self.start]
        class Detector:
            def detect(self, *args, **kwargs):
                return []
        class Policy:
            calls = 0
            def decide(self, *args):
                return dict(action='explore', candidate_id=None)
        with tempfile.TemporaryDirectory() as tmp:
            result = VisualNavigator(IO(), Detector(), Policy(), tmp, max_observations=40).run('chair')
            self.assertEqual(result['status'], 'search_complete_no_match')
            self.assertEqual(result['observations'], 4)


class MemoryTests(unittest.TestCase):
    def test_multiview_merge_but_same_frame_instances_stay_distinct(self):
        memory = ObjectMemory()
        ids = memory.integrate([detection(0), detection(.4)], observation())
        self.assertEqual(len(set(ids)), 2)
        memory.integrate([detection(.03)], observation('o2'))
        self.assertEqual(len(memory.instances), 2)
        self.assertEqual(memory.instances['v0001']['views'], 2)

    def test_no_detection_does_not_erase_and_color_can_change(self):
        memory = ObjectMemory()
        memory.integrate([detection()], observation())
        memory.integrate([], observation('o2'))
        self.assertEqual(memory.instances['v0001']['last_seen'], 'o1')
        self.assertFalse(memory.rejections)
        for i in range(3,7):
            memory.integrate([detection(color='white')], observation('o'+str(i)))
        scores = memory.instances['v0001']['attributes']['color_scores']
        self.assertGreater(scores['white'], scores['red'])

    def test_rejection_requires_fresh_source(self):
        memory = ObjectMemory()
        memory.integrate([detection()], observation())
        with self.assertRaises(ValueError):
            memory.reject('v0001', 'o2', 'not seen')
        memory.reject('v0001', 'o1', 'visible mismatch')
        self.assertEqual(memory.rejections[0]['observation_id'], 'o1')


class DecisionTests(unittest.TestCase):
    def test_multimodal_request_uses_only_whitelisted_visual_context(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frame.jpg'
            Image.new('RGB', (32,32), 'white').save(path)
            obs = observation()
            obs['rgb_path'] = str(path)
            # A future upstream accidental extra field must not leak into the prompt.
            obs['secret_gt_table'] = 'DO_NOT_SEND'
            memory = ObjectMemory()
            d = detection()
            d['crop'] = str(path)
            memory.integrate([d], obs)
            config = dict(base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', api_key='test', timeout_seconds=2)
            packet = {'choices':[{'finish_reason':'stop', 'message':{'content':json.dumps(
                dict(action='select', candidate_id='v0001'))}}]}
            with patch('model_settings.load_model_settings', return_value=config), patch.dict(
                    'os.environ', {'NAV_VLM_MODEL':'qwen-test', 'NAV_VLM_BASE_URL':config['base_url'], 'NAV_VLM_API_KEY':'test'}), patch(
                    'visual_nav.policy.request.urlopen', return_value=io.BytesIO(json.dumps(packet).encode())) as call:
                policy = VLMPolicy()
                policy.decide('chair', obs, memory)
                payload = json.loads(call.call_args[0][0].data)
                self.assertNotIn('DO_NOT_SEND', json.dumps(payload))
                content = payload['messages'][1]['content']
                self.assertEqual(sum(item['type'] == 'image_url' for item in content), 2)
                self.assertEqual(payload['response_format'], {'type':'json_object'})

    def test_confirmation_needs_fresh_selected_visible_close_candidate(self):
        decision = dict(action='confirm', candidate_id='v1', evidence_id='o2')
        validate_decision(decision, {'v1'}, 'o2', ['v1'], 'v1', True)
        for current, seen, selected, at_goal in [('o3',['v1'],'v1',True), ('o2',[],'v1',True),
                                                ('o2',['v1'],'v2',True), ('o2',['v1'],'v1',False)]:
            with self.assertRaises(ValueError):
                validate_decision(decision, {'v1'}, current, seen, selected, at_goal)

    def test_no_fabricated_identity_or_path(self):
        for d in [dict(action='select', candidate_id='chair_gt_1'),
                  dict(action='select', candidate_id='v1', path=[[0,0,0]])]:
            with self.assertRaises(ValueError):
                validate_decision(d, {'v1'}, 'o1', ['v1'], None, False)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.objects = [dict(id='a', bbox_min=[-.3,.5,-2.3], bbox_max=[.3,1.1,-1.7]),
                        dict(id='b', bbox_min=[.7,.5,-2.3], bbox_max=[1.3,1.1,-1.7])]

    def test_correct_identity_wrong_color_and_ambiguous_are_distinct(self):
        points = detection()['points']
        self.assertEqual(associate(points, self.objects)[0], 'a')
        duplicate = dict(self.objects[0], id='duplicate')
        self.assertIsNone(associate(points, self.objects+[duplicate])[0])
        result = dict(mode='active_verify', status='succeeded', selected_points=points,
                      final_position=[0,0,-1], path_length_m=1, observations=3, elapsed_seconds=2)
        episode = dict(id='test', goal_ids=['b'])
        row = score_episode(result, episode, self.objects)
        self.assertTrue(row['wrong_target'])
        self.assertFalse(row['success'])
        ambiguous_wrong = score_episode(result, episode, self.objects+[duplicate])
        self.assertTrue(ambiguous_wrong['wrong_target'])
        self.assertTrue(ambiguous_wrong['unresolved_identity'])
        episode['goal_ids'] = ['a']
        self.assertTrue(score_episode(result, episode, self.objects)['success'])

    def test_negative_episode_not_counted_as_navigation_success(self):
        row = dict(mode='active_verify', policy='test', goal_present=False, success=False,
                   wrong_target=False, unresolved_identity=False, abstained=True,
                   path_length_m=0, observations=2, elapsed_seconds=1)
        group = aggregate([row])['active_verify/test']
        self.assertIsNone(group['success_rate'])
        self.assertEqual(group['negative_abstention_rate'], 1)
        row['status'] = 'error'
        failed = aggregate([row])['active_verify/test']
        self.assertEqual(failed['negative_abstention_rate'], 0)
        self.assertEqual(failed['error_rate'], 1)


class PermissionTests(unittest.TestCase):
    def test_policy_and_perception_do_not_import_gt_modules(self):
        root = Path(__file__).parent
        for name in ('loop', 'memory', 'policy', 'perception', 'geometry', 'roadmap'):
            source = (root/'visual_nav'/(name+'.py')).read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, {'world', 'session', 'planning', 'evaluate_visual', 'serve'})
            for forbidden in ('annotations.json', 'semantic_id', 'build_map(', 'get_rigid_object_manager'):
                self.assertNotIn(forbidden, source)


class LoopTests(unittest.TestCase):
    def test_arrival_requires_new_observation_not_initial_confidence(self):
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
        class Detector:
            def detect(self, obs, output, categories):
                return [detection()]
        class Policy:
            calls = 0
            def decide(self, instruction, obs, memory, selected, at_goal):
                return dict(action='confirm' if at_goal else 'select', candidate_id='v0001', evidence_id=obs['id'])
        with tempfile.TemporaryDirectory() as tmp:
            result = VisualNavigator(IO(), Detector(), Policy(), tmp, mode='verify').run('chair')
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(result['observations'], 2)
            self.assertEqual(result['events'][-1]['evidence_id'], 'o2')
            class FarIO(IO):
                def docking_options(self, instance):
                    return [np.array([0,0,-1.])]
            far = VisualNavigator(FarIO(), Detector(), Policy(), tmp, mode='verify', max_observations=2).run('chair')
            self.assertFalse(far['verified'])
            self.assertNotEqual(far['status'], 'succeeded')
            self.assertGreater(far['arrival_geometry']['estimated_boundary_gap_m'], .65)
            class StalledIO(IO):
                def advance(self, endpoint, **kwargs):
                    return dict(status='succeeded', reached=False)
            stalled = VisualNavigator(StalledIO(), Detector(), Policy(), tmp, mode='candidates').run('chair')
            self.assertEqual(stalled['status'], 'timeout')
            self.assertFalse(stalled['verified'])



class GeneralCategoryTests(unittest.TestCase):
    def test_task_vocabulary_is_scene_independent_and_open(self):
        from visual_nav.task import validate_task, COMMON_CATEGORIES
        task = validate_task(dict(supported=True, target_categories=['microwave'],
                                  reference_categories=['fridge'], attributes=['red'], relation='near'))
        self.assertEqual(task['target_categories'], ['microwave'])
        self.assertEqual(task['reference_categories'], ['refrigerator'])
        self.assertEqual(task['detection_categories'][:2], ['microwave', 'refrigerator'])
        self.assertTrue(set(COMMON_CATEGORIES).issubset(task['detection_categories']))
        with self.assertRaises(ValueError):
            validate_task(dict(supported=True, target_categories=['chair. sofa.']))
        with self.assertRaises(ValueError):
            validate_task(dict(supported=False, reason='multi-step'))

    def test_instruction_parser_receives_only_instruction(self):
        from visual_nav.policy import VLMPolicy
        policy = object.__new__(VLMPolicy)
        parsed = dict(supported=True, target_categories=['printer'], attributes=['white'])
        with patch.object(policy, '_request', return_value=parsed) as request:
            task = policy.parse_instruction('走到白色打印机旁边')
        self.assertEqual(request.call_args.args[1], '走到白色打印机旁边')
        self.assertEqual(task['target_categories'], ['printer'])

    def test_simple_go_beside_object_is_target_not_missing_reference(self):
        from visual_nav.task import simple_navigation_task
        for text in ('去棕色沙发旁边', '去沙发', '走到沙发附近'):
            task = simple_navigation_task(text)
            self.assertEqual(task['target_categories'], ['sofa'])
            self.assertEqual(task['reference_categories'], [])
        self.assertEqual(simple_navigation_task('去棕色沙发旁边')['attributes'], ['brown'])
        for text in ('先去沙发再去桌子', '去桌旁的红色椅子', '打开冰箱', '去打印机旁边'):
            self.assertIsNone(simple_navigation_task(text))
        policy = object.__new__(VLMPolicy)
        with patch.object(policy, '_request') as request:
            self.assertEqual(policy.parse_instruction('去棕色沙发旁边')['target_categories'], ['sofa'])
            request.assert_not_called()

    def test_non_chair_observation_populates_memory(self):
        from visual_nav.task import validate_task
        class Camera:
            def observe(self, **kwargs):
                return observation()
        class Detector:
            def detect(self, obs, output, categories):
                self.categories = categories
                d = detection()
                d['category'] = 'cabinet'
                return [d]
        detector = Detector()
        task = validate_task(dict(supported=True, target_categories=['cabinet']))
        with tempfile.TemporaryDirectory() as tmp:
            nav = VisualNavigator(Camera(), detector, None, tmp, task=task)
            nav.observe()
        self.assertEqual(detector.categories[0], 'cabinet')
        self.assertIn('sofa', detector.categories)
        self.assertEqual(nav.memory.instances['v0001']['category'], 'cabinet')

class DockingTests(unittest.TestCase):
    def test_far_waypoint_is_not_arrival_and_boundary_not_center_controls_large_sofa(self):
        from visual_nav.docking import arrival_geometry, boundary_samples
        # Large sofa: a valid side stop is far from its center but near its surface.
        instance = dict(bbox_min=[0,0,0], bbox_max=[1,1,3],
                        points=[[0,.5,z] for z in np.linspace(0,3,100)])
        self.assertFalse(arrival_geometry([-1.04,0,1.5],instance)['close_enough'])
        self.assertTrue(arrival_geometry([-.5,0,1.5],instance)['close_enough'])
        self.assertFalse(arrival_geometry([-.1,0,1.5],instance)['close_enough'])
        self.assertTrue(any(arrival_geometry(p,instance)['close_enough'] for p in boundary_samples(instance,0)))
        # A huge bbox by itself is insufficient: there must be nearby observed surface support.
        instance['points']=[[1,.5,z] for z in np.linspace(0,3,100)]
        self.assertFalse(arrival_geometry([-.5,0,1.5],instance)['close_enough'])

if __name__ == '__main__':
    unittest.main()
