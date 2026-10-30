"""Trusted simulator adapter. Exposes RGB-D/poses and geometric navigation only.

Simulator internals necessarily load asset metadata. None is returned to the policy.
Do not call World.objects/build_map, annotations, semantic sensors, or GT UI here.
"""
import math
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import horizontal_distance, path_length
from .docking import arrival_geometry, boundary_samples, PREFERRED_GAP


class HabitatIO:
    def __init__(self, scene, output, seed=7, start=None, frame_callback=None, realtime=False):
        from world import World
        from executor import Executor
        class GeometryWorld(World):
            def objects(self, *args, **kwargs):
                raise PermissionError('GT object access is forbidden in visual mode')

            def build_map(self, *args, **kwargs):
                raise PermissionError('GT-derived roadmap is forbidden in visual mode')
        self._world = GeometryWorld(scene=scene, seed=seed, rgbd=True)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.counter, self.yaw = 0, 0.0
        pf = self._world.pf
        if start is None:
            start = pf.get_random_navigable_point(island_index=self._world.island)
        snapped = np.asarray(pf.snap_point(start, self._world.island))
        if not np.isfinite(snapped).all() or np.linalg.norm(snapped-start) > .2:
            self._world.close()
            raise ValueError('Start must be on the main navigable island')
        self._executor = Executor(self._world, snapped, frame_callback=frame_callback, realtime=realtime)
        self.start = self.position.tolist()
        self.visited = [self.start]
        self.last_motion = None
        self._roadmap = None
        self._search = None
        self.should_stop = lambda: False
        if frame_callback:
            frame_callback(self._world.frame(self.position))

    def _search_state(self):
        if self._search is None:
            from .search import CoverageSearch
            bounds = self._world.pf.get_bounds()
            points = [self.position.copy()]
            for x in np.arange(bounds[0][0], bounds[1][0], .6):
                for z in np.arange(bounds[0][2], bounds[1][2], .6):
                    query = np.array([x, self.position[1], z])
                    point = np.asarray(self._world.pf.snap_point(query, self._world.island))
                    if np.isfinite(point).all() and np.linalg.norm(point-query) < .15:
                        points.append(point)
            def route_distance(point):
                path = self.route(point)
                return path_length(path) if path is not None else None
            self._search = CoverageSearch(points, self._world.direct, route_distance=route_distance)
        return self._search

    def search_summary(self):
        return self._search_state().summary()

    def search_report(self):
        state = self._search_state()
        return dict(summary=state.summary(), scans=state.scans,
                    unresolved_points=[state.points[i].tolist() for i in sorted(state.blocked)])

    def mark_search_failure(self, goal):
        self._search_state().failed(goal)

    @property
    def position(self):
        return self._executor.position

    def settle(self):
        self._executor.stop()

    def close(self):
        self._world.close()

    def begin_task(self, output, should_stop=None):
        """Keep the simulator/body/observation IDs; reset only task-local accounting."""
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.start = self.position.tolist()
        self._executor.trace.clear()
        self._executor.trace.append(self.start)
        self.visited = [self.start]
        self.last_motion = None
        # Coverage is instruction-dependent: old views may not have detected the new noun.
        self._search = None
        self.should_stop = should_stop or (lambda: False)

    def observe(self, yaw=None, look_at=None):
        import magnum as mn
        if look_at is not None:
            delta = np.asarray(look_at)-self.position
            yaw = math.atan2(delta[0], -delta[2])
        if yaw is not None:
            self.yaw = float(yaw)
        forward = np.array([math.sin(self.yaw), 0, -math.cos(self.yaw)])
        eye = self.position + [0, 1.65, 0]
        aim = np.asarray(look_at) if look_at is not None else eye + forward + [0, -.22, 0]
        if np.linalg.norm(aim-eye) < .1:
            aim = eye + forward
        agent = self._world.sim.get_agent(0)
        agent.scene_node.transformation = mn.Matrix4.look_at(mn.Vector3(eye), mn.Vector3(aim), mn.Vector3.y_axis())
        # Hide only our own body during egocentric sensing (not scene furniture).
        human = self._executor.human.sim_obj
        body_pose = mn.Matrix4(human.transformation)
        try:
            # Habitat 0.3.3 articulated objects have no is_visible property.
            # Temporarily move only this body outside the render volume; no physics step.
            human.translation = mn.Vector3(0, -1000, 0)
            raw = self._world.sim.get_sensor_observations()
        finally:
            human.transformation = body_pose
        rgb = raw['rgb'][:, :, :3].copy()
        depth = np.asarray(raw['depth']).copy()
        sensor = agent._sensors['rgb']
        pose = np.array(sensor.node.absolute_transformation(), dtype=float)
        h, w = depth.shape
        hfov = float(sensor.specification().hfov)
        f = (w/2)/math.tan(math.radians(hfov)/2)
        k = np.array([[f, 0, (w-1)/2], [0, f, (h-1)/2], [0, 0, 1]])
        self.counter += 1
        key = 'obs_%05d' % self.counter
        rgb_path = self.output/(key+'.jpg')
        Image.fromarray(rgb).save(rgb_path, quality=95)
        np.savez_compressed(self.output/(key+'.npz'), depth=depth, intrinsics=k, camera_to_world=pose)
        observation = dict(id=key, rgb=rgb, depth=depth, intrinsics=k, camera_to_world=pose,
                    rgb_path=str(rgb_path), time=time.time(), position=self.position.tolist(), yaw=self.yaw)
        if look_at is None:
            self._search_state().record_view(self.position, self.yaw, key, observation)
        return observation

    def route(self, endpoint):
        import habitat_sim
        path = habitat_sim.ShortestPath()
        path.requested_start = self.position
        path.requested_end = np.asarray(endpoint)
        if not self._world.pf.find_path(path):
            return None
        return [np.asarray(p, dtype=float) for p in path.points]

    def calibration_check(self, observation):
        """Diagnostic only: RGB-D projection versus geometry rays, with no semantic output."""
        import habitat_sim
        import magnum as mn
        k, pose = observation['intrinsics'], observation['camera_to_world']
        depth = observation['depth']
        errors = []
        for v in range(64, depth.shape[0]-32, 64):
            for u in range(64, depth.shape[1]-32, 64):
                d = float(depth[v, u])
                if not .15 < d < 8:
                    continue
                local = np.array([(u-k[0,2])/k[0,0], -(v-k[1,2])/k[1,1], -1.])
                ray = pose[:3,:3] @ local
                hits = self._world.sim.cast_ray(habitat_sim.geo.Ray(
                    mn.Vector3(pose[:3,3]), mn.Vector3(ray/np.linalg.norm(ray)))).hits
                distances = [h.ray_distance for h in hits
                             if h.object_id != self._world.human_id and h.ray_distance > .05]
                if distances:
                    errors.append(abs(min(distances)-d*np.linalg.norm(local)))
        return {'rays': len(errors), 'median_error_m': float(np.median(errors)) if errors else None,
                'p90_error_m': float(np.quantile(errors, .9)) if errors else None}

    def docking_options(self, instance):
        pf = self._world.pf
        options = []
        used = set()
        for query in boundary_samples(instance, self.position[1]):
            point = np.asarray(pf.snap_point(query, self._world.island))
            if not np.isfinite(point).all() or np.linalg.norm(point-query) > .15:
                continue
            key = tuple(np.round(point, 2))
            if key in used:
                continue
            used.add(key)
            geometry = arrival_geometry(point, instance)
            if not geometry['close_enough']:
                continue
            path = self.route(point)
            if path is not None:
                # First prefer the desired clearance; use path length within that band.
                band = round(abs(geometry['estimated_boundary_gap_m']-PREFERRED_GAP), 1)
                options.append((band, path_length(path), point))
        return [point for _, _, point in sorted(options, key=lambda x: (x[0], x[1]))]

    def inspection_options(self, instance, reference=None, historical=(), visited=()):
        from .inspection import joint_aim, rank_viewpoints
        import habitat_sim
        import magnum as mn
        center = joint_aim(instance, reference)
        queries = [np.asarray(v['position']) for v in historical]
        queries += self.docking_options(instance)
        for radius in (1., 1.8, 2.5):
            for angle in np.linspace(0, 2*math.pi, 16, endpoint=False):
                queries.append(np.array([center[0]+radius*math.cos(angle), self.position[1],
                                         center[2]+radius*math.sin(angle)]))
        points, used = [], set()
        for query in queries:
            point = np.asarray(self._world.pf.snap_point(query, self._world.island))
            key = tuple(np.round(point, 2))
            if np.isfinite(point).all() and np.linalg.norm(point-query) < .15 and key not in used:
                used.add(key)
                points.append(point)
        def visible(eye, item):
            ray = np.asarray(item['position'])-eye
            distance = np.linalg.norm(ray)
            if distance < .15:
                return False
            hits = self._world.sim.cast_ray(habitat_sim.geo.Ray(
                mn.Vector3(eye), mn.Vector3(ray/distance))).hits
            distances = [h.ray_distance for h in hits if h.object_id != self._world.human_id and h.ray_distance > .05]
            extent = np.linalg.norm(np.asarray(item['bbox_max'])-item['bbox_min'])/2
            return not distances or min(distances) >= distance-extent-.15
        return rank_viewpoints(points, self.position, instance, reference, self.route, visible, visited)

    def advance(self, endpoint, budget=1.0, on_step=None):
        """Travel <= budget metres along NavMesh path, checking every controller step."""
        route = self._roadmap.path(self.position, endpoint) if self._roadmap else self.route(endpoint)
        if route is None:
            return dict(status='unreachable', position=self.position.tolist(), travelled=0)
        before = len(self._executor.trace)
        remaining = budget
        event = dict(status='succeeded')
        pending = list(route[1:])
        segments = 0
        while pending and segments < 80:
            point = pending.pop(0)
            segments += 1
            delta = point-self.position
            length = np.linalg.norm(delta)
            if length < .02:
                continue
            target = self.position + delta*min(1, remaining/length)
            fallback = False
            if not self._world.direct(self.position, target):
                # Shortest-path corners can sit exactly on a boundary; finite controller
                # arrival error makes the next edge non-traversable. Use interior samples.
                if self._roadmap is None:
                    from .roadmap import GeometricRoadmap
                    self._roadmap = GeometricRoadmap(self._world.pf, self._world.island,
                                                     self._world.direct, self.position[1])
                alternative = self._roadmap.path(self.position, endpoint)
                if alternative is None or len(alternative) < 2:
                    event = dict(status='blocked', reason='No conservative geometry route')
                    break
                delta = alternative[1]-self.position
                length = np.linalg.norm(delta)
                target = self.position + delta*min(1, remaining/max(length, 1e-6))
                fallback = True
                if on_step is not None:
                    pending = list(alternative[2:])
            self.yaw = math.atan2(delta[0], -delta[2])
            previous = self.position.copy()
            event = self._executor.execute_segment(target, arrival_tolerance=.02, should_stop=self.should_stop,
                on_step=on_step, settle=on_step is None)
            event['waypoint'] = target.tolist()
            remaining -= np.linalg.norm(self.position-previous)
            if event['status'] != 'succeeded' or remaining < .10 or (fallback and on_step is None):
                break
        if on_step is not None:
            self._executor.stop()
        self.visited.append(self.position.tolist())
        self.last_motion = dict(event, travelled=path_length(list(self._executor.trace)[max(0,before-1):]),
                                position=self.position.tolist(), reached=horizontal_distance(self.position, endpoint) < .15)
        return self.last_motion

    def explore_goal(self):
        return self._search_state().next_goal(self.position)

    def search_yaw(self):
        return self._search_state().next_view(self.position)

    def trajectory(self):
        return list(self._executor.trace)
