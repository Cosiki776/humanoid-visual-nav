"""ReplicaCAD GT and a conservative, direct-edge roadmap (no path search)."""
import json
from itertools import product

import bootstrap
import habitat_sim
import magnum as mn
import numpy as np


def vector(value):
    return [round(float(v), 5) for v in value]


class World:
    def __init__(self, scene="apt_0", radius=0.3, seed=7, rgbd=False):
        self.data = bootstrap.DATA / "replica_cad"
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_dataset_config_file = str(self.data / "replicaCAD.scene_dataset_config.json")
        cfg.scene_id = scene
        cfg.enable_physics = True
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "rgb"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [512, 768]
        sensor.position = mn.Vector3(0.0, 0.0, 0.0)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [sensor]
        if rgbd:
            # Opt-in keeps the original GT demo's sensor configuration unchanged.
            depth = habitat_sim.CameraSensorSpec()
            depth.uuid = "depth"
            depth.sensor_type = habitat_sim.SensorType.DEPTH
            depth.resolution = list(sensor.resolution)
            depth.position = mn.Vector3(0.0, 0.0, 0.0)
            depth.hfov = sensor.hfov
            agent.sensor_specifications.append(depth)
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
        # Static demo: freeze clutter before rebuilding a body-size-aware NavMesh.
        for obj in self.sim.get_rigid_object_manager().get_objects_by_handle_substring().values():
            obj.motion_type = habitat_sim.physics.MotionType.STATIC
        for obj in self.sim.get_articulated_object_manager().get_objects_by_handle_substring().values():
            obj.motion_type = habitat_sim.physics.MotionType.STATIC
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = radius
        settings.agent_height = 1.8
        settings.include_static_objects = True
        if not self.sim.recompute_navmesh(self.sim.pathfinder, settings):
            raise RuntimeError("NavMesh build failed")
        self.pf = self.sim.pathfinder
        self.pf.seed(seed)
        self.island = max(range(self.pf.num_islands), key=self.pf.island_area)
        self.radius = radius
        self.scene = scene

    def close(self):
        self.sim.close()

    def direct(self, start, end):
        """Require straight traversal in both directions; never find a detour."""
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if not np.isfinite([a, b]).all() or abs(a[1] - b[1]) > 0.15:
            return False
        for p, q in ((a, b), (b, a)):
            if np.linalg.norm(np.asarray(self.pf.try_step_no_sliding(p, q)) - q) > 0.025:
                return False
            for t in np.linspace(0, 1, max(2, int(np.linalg.norm(q-p)/0.08)+1)):
                if not self.pf.is_navigable(p + t * (q-p), 0.08):
                    return False
        return True

    def objects(self, annotations=None):
        lexicon = json.loads((self.data / "configs/ssd/replicaCAD_semantic_lexicon.json").read_text())
        labels = {v["id"]: v["name"] for v in lexicon["classes"]}
        attrs = annotations or {}
        result = []
        objects = self.sim.get_rigid_object_manager().get_objects_by_handle_substring()
        for handle, obj in sorted(objects.items()):
            template = obj.creation_attributes.handle
            name = template.split("/")[-1].replace(".object_config.json", "")
            config_path = self.data / "configs/objects" / (name + ".object_config.json")
            meta = json.loads(config_path.read_text()) if config_path.exists() else {}
            bb = obj.root_scene_node.cumulative_bb
            corners = [obj.transformation.transform_point(mn.Vector3(x, y, z))
                       for x, y, z in product(*zip(bb.min, bb.max))]
            corners = np.asarray(corners)
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            # Stable per loaded scene; runtime IDs are kept out of the language interface.
            result.append({"id": handle, "category": labels.get(meta.get("semantic_id"), "unknown"),
                           "position_world": vector(obj.translation),
                           "bbox_min": vector(lo), "bbox_max": vector(hi),
                           "attributes": attrs.get(handle, {}), "goal_nodes": []})
        return result

    def build_map(self, annotations=None, spacing=0.8):
        nodes = {}
        bounds = self.pf.get_bounds()
        floor = float(self.pf.get_random_navigable_point(island_index=self.island)[1])
        for x in np.arange(bounds[0][0], bounds[1][0], spacing):
            for z in np.arange(bounds[0][2], bounds[1][2], spacing):
                p = np.asarray(self.pf.snap_point([x, floor, z], self.island))
                if not np.isfinite(p).all() or np.linalg.norm(p - [x, floor, z]) > 0.15:
                    continue
                nodes[f"n{len(nodes):03d}"] = vector(p)
        objects = self.objects(annotations)
        # Navigation targets are furniture; clutter remains in the GT object list.
        for obj in objects:
            if obj["category"] not in {"chair", "sofa", "table", "bed", "stool"}:
                continue
            lo, hi = np.array(obj["bbox_min"]), np.array(obj["bbox_max"])
            center = (lo + hi) / 2
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                d = np.array([np.cos(angle), 0, np.sin(angle)])
                half = (hi-lo)/2
                extents = [half[i]/abs(d[i]) for i in (0, 2) if abs(d[i]) > 1e-6]
                p = center + d*(min(extents) + self.radius + 0.3)
                p[1] = floor
                snapped = np.asarray(self.pf.snap_point(p, self.island))
                if not np.isfinite(snapped).all() or np.linalg.norm(snapped-p) > 0.2:
                    continue
                # Distance to horizontal bbox, not object center.
                gap = np.maximum(np.maximum(lo[[0, 2]]-snapped[[0, 2]],
                                            snapped[[0, 2]]-hi[[0, 2]]), 0)
                if not self.radius + 0.1 <= np.linalg.norm(gap) <= 1.0:
                    continue
                key = f"n{len(nodes):03d}"
                nodes[key] = vector(snapped)
                obj["goal_nodes"].append(key)
        edges = []
        keys = list(nodes)
        for i, a in enumerate(keys):
            for b in keys[i+1:]:
                distance = np.linalg.norm(np.array(nodes[a])-nodes[b])
                if 0.01 < distance <= 1.65 and self.direct(nodes[a], nodes[b]):
                    edges.append([a, b])
        # Keep largest graph component. A NavMesh island can be undersampled by a roadmap.
        adjacency = {k: set() for k in keys}
        for a, b in edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        remaining, components = set(keys), []
        while remaining:
            todo = [min(remaining)]
            component = set(todo)
            while todo:
                for nxt in adjacency[todo.pop()] - component:
                    component.add(nxt)
                    todo.append(nxt)
            remaining -= component
            components.append(component)
        if not components:
            raise RuntimeError("No roadmap nodes; check scene/NavMesh")
        keep = max(components, key=len)
        nodes = {k: v for k, v in nodes.items() if k in keep}
        for obj in objects:
            obj["goal_nodes"] = [k for k in obj["goal_nodes"] if k in keep]
        return {"scene": self.scene, "coordinate_frame": "Habitat world, meters, Y up",
                "agent_radius_m": self.radius, "nodes": nodes,
                "edges": [e for e in edges if set(e) <= keep], "objects": objects}

    def frame(self, position):
        p = mn.Vector3(position)
        aim = p + mn.Vector3(0, 1.0, 0)
        candidates = []
        for distance in (2.2, 1.5, 1.0):
            for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                eye = p + mn.Vector3(distance*np.cos(angle), 1.9, distance*np.sin(angle))
                delta = eye-aim
                hits = self.sim.cast_ray(habitat_sim.geo.Ray(aim, delta.normalized())).hits
                occluded = any(h.object_id != getattr(self, "human_id", None)
                               and 0.05 < h.ray_distance < delta.length()+0.1 for h in hits)
                if not occluded:
                    candidates.append(eye)
            if candidates:
                break
        if candidates:
            old = getattr(self, "camera_offset", mn.Vector3(2.0, 1.9, 2.0))
            eye = min(candidates, key=lambda e: (e-p-old).length())
            self.camera_offset = eye-p
        else:
            eye = p + mn.Vector3(0.3, 1.9, 0.3)
        camera = self.sim.get_agent(0).scene_node
        camera.transformation = mn.Matrix4.look_at(eye, aim, mn.Vector3.y_axis())
        return self.sim.get_sensor_observations()["rgb"][:, :, :3]
