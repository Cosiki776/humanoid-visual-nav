"""Track supplied segments with Habitat's kinematic humanoid controller."""
from types import SimpleNamespace
from collections import deque
import time

import bootstrap
import magnum as mn
import numpy as np
from habitat.articulated_agents.humanoids import KinematicHumanoid
from habitat.articulated_agent_controllers import HumanoidRearrangeController
from world import vector


class _LabSimulatorAdapter:
    """Lab's humanoid expects a sensor registry; the demo uses an external camera."""
    sensors = {}

    def __init__(self, sim):
        self.sim = sim

    def __getattr__(self, name):
        return getattr(self.sim, name)


class Executor:
    def __init__(self, world, start, frame_callback=None, realtime=False, trace_limit=None):
        assets = bootstrap.ROOT / "assets/female_0"
        motion = assets / "female_0_motion_data_smplx.pkl"
        cfg = SimpleNamespace(auto_update_sensor_transform=False,
                              articulated_agent_urdf=str(assets / "female_0.urdf"),
                              motion_data_path=str(motion))
        self.world = world
        self.human = KinematicHumanoid(cfg, _LabSimulatorAdapter(world.sim))
        self.human.reconfigure()
        world.human_id = self.human.sim_obj.object_id
        self.human.base_pos = mn.Vector3(start)
        self.controller = HumanoidRearrangeController(str(motion))
        self.controller.reset(self.human.base_transformation)
        self.controller.set_framerate_for_linspeed(0.8, 1.5, 30)
        self.frame_callback = frame_callback
        self.trace = ([vector(self.human.base_pos)] if trace_limit is None else
                      deque([vector(self.human.base_pos)], maxlen=trace_limit))
        self.realtime = realtime
        self.last_tick = time.monotonic()
        self.frames = 0
        self.stop()

    @property
    def position(self):
        return np.asarray(self.human.base_pos).copy()

    def apply(self, force_frame=False):
        if self.realtime:
            time.sleep(max(0, 1/30 - (time.monotonic()-self.last_tick)))
            self.last_tick = time.monotonic()
        self.human.set_joint_transform(self.controller.joint_pose,
                                       self.controller.obj_transform_offset,
                                       self.controller.obj_transform_base)
        self.world.sim.step_physics(1/30)
        self.trace.append(vector(self.position))
        self.frames += 1
        if self.frame_callback and (force_frame or self.frames % 2 == 0):
            self.frame_callback(self.world.frame(self.position))

    def stop(self):
        self.controller.calculate_stop_pose()
        # Always publish the settled pose, including odd-numbered final steps.
        self.apply(force_frame=True)

    def execute_segment(self, endpoint, max_steps=600, should_stop=None, arrival_tolerance=0.08):
        if not 0.005 <= arrival_tolerance <= 0.08:
            raise ValueError("arrival_tolerance must be 0.005..0.08 metres")
        target = np.array(endpoint, dtype=float)
        if not self.world.direct(self.position, target):
            return {"status": "blocked", "reason": "segment is not directly traversable",
                    "position": vector(self.position)}
        stalled = 0
        for step in range(max_steps):
            if should_stop and should_stop():
                self.stop()
                return {"status": "cancelled", "position": vector(self.position)}
            before = self.position
            if np.linalg.norm(target-before) < arrival_tolerance:
                self.stop()
                return {"status": "succeeded", "steps": step, "position": vector(self.position)}
            self.controller.calculate_walk_pose(mn.Vector3(target-before))
            candidate = np.asarray(self.controller.obj_transform_base.translation) - [0, 0.9, 0]
            filtered = np.asarray(self.world.pf.try_step_no_sliding(before, candidate))
            if not np.isfinite(filtered).all() or np.linalg.norm(filtered-candidate) > 0.03:
                self.controller.reset(self.human.base_transformation)
                self.stop()
                return {"status": "blocked", "reason": "NavMesh rejected controller step",
                        "position": vector(self.position)}
            self.controller.obj_transform_base.translation = mn.Vector3(filtered + [0, 0.9, 0])
            self.apply()
            stalled = stalled + 1 if np.linalg.norm(self.position-before) < 1e-5 else 0
            if stalled > 150:
                self.stop()
                return {"status": "blocked", "reason": "no progress", "position": vector(self.position)}
        self.stop()
        return {"status": "timeout", "position": vector(self.position)}
