"""Bounded visual snapshots, indexed by observed objects and camera pose only."""
import copy
import math
import numpy as np


class ViewMemory:
    def __init__(self, capacity=48):
        self.capacity = capacity
        self.views = []

    def add(self, observation, seen):
        pose = np.asarray(observation['camera_to_world'])
        forward = -pose[:3, 2]
        yaw = float(math.atan2(forward[0], -forward[2]))
        item = dict(id=observation['id'], rgb=observation['rgb_path'],
                    position=list(observation.get('position', pose[:3, 3].tolist())),
                    camera_position=pose[:3, 3].tolist(), yaw=yaw,
                    time=observation['time'], seen=list(seen))
        replacement = None
        for index, old in enumerate(self.views):
            angle = abs(math.atan2(math.sin(yaw-old['yaw']), math.cos(yaw-old['yaw'])))
            if np.linalg.norm(pose[:3, 3]-old['camera_position']) < .6 and angle < math.radians(25):
                # Keep distinct evidence when an object disappears behind an occluder.
                if set(old['seen']) <= set(seen):
                    replacement = index
                    break
        if replacement is not None:
            self.views.pop(replacement)
        self.views.append(item)
        self.views = self.views[-self.capacity:]

    def retrieve(self, ids, exclude=(), limit=1):
        ids, exclude = set(ids), set(exclude)
        ranked = [v for v in self.views if v['id'] not in exclude and ids.intersection(v['seen'])]
        ranked.sort(key=lambda v: (len(ids.intersection(v['seen'])), v['time']), reverse=True)
        return copy.deepcopy(ranked[:limit])

    def summary(self):
        return copy.deepcopy(self.views)
