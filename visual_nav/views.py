"""One frame index, stationary observation groups, and derived task context."""
import copy
import math
import numpy as np


def angle_delta(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


class ViewMemory:
    def __init__(self, capacity=48):
        self.capacity = capacity
        self.views = []
        # Groups reference frames; neither pixels nor pose records are duplicated.
        self.groups = []
        self.next_group = 1

    def add(self, observation, seen):
        pose = np.asarray(observation['camera_to_world'])
        forward = -pose[:3, 2]
        yaw = float(math.atan2(forward[0], -forward[2]))
        position = list(observation.get('position', pose[:3, 3].tolist()))
        kind = observation.get('capture_kind', 'keyframe')
        item = dict(id=observation['id'], rgb=observation['rgb_path'], position=position,
                    camera_position=pose[:3, 3].tolist(), camera_to_world=pose.tolist(),
                    yaw=yaw, time=observation['time'], seen=list(seen), capture_kind=kind)
        group = self.groups[-1] if self.groups else None
        stationary = (kind == 'scan' and group and group['kind'] == 'scan' and
                      np.linalg.norm(np.asarray(position)-group['position']) <= .15)
        if not stationary:
            group = dict(id='view_group_%04d' % self.next_group, kind=kind,
                         position=position, yaw=yaw, frame_ids=[])
            self.next_group += 1
            self.groups.append(group)
        item['group_id'] = group['id']
        # Keep the latest observation in a repeated direction, without claiming
        # old detections remain visible. Object evidence retains older sources.
        replaced = []
        if kind == 'scan':
            replaced = [v['id'] for v in self.views if v['id'] in group['frame_ids'] and
                        abs(angle_delta(v['yaw'], yaw)) < math.radians(20)]
        if kind == 'keyframe':
            replaced = [v['id'] for v in self.views if v['capture_kind'] == 'keyframe' and
                        np.linalg.norm(pose[:3, 3]-v['camera_position']) < .15 and
                        abs(angle_delta(v['yaw'], yaw)) < math.radians(20) and set(v['seen']) <= set(seen)]
        self.views = [v for v in self.views if v['id'] not in replaced and v['id'] != item['id']]
        group['frame_ids'] = [key for key in group['frame_ids'] if key not in replaced and key != item['id']]
        group['frame_ids'].append(item['id'])
        self.views.append(item)
        self.views = self.views[-self.capacity:]
        retained = {v['id'] for v in self.views}
        for g in self.groups:
            g['frame_ids'] = [key for key in g['frame_ids'] if key in retained]
        self.groups = [g for g in self.groups if g['frame_ids']]

    def retrieve(self, ids, exclude=(), limit=1):
        ids, exclude = set(ids), set(exclude)
        ranked = [v for v in self.views if v['id'] not in exclude and ids.intersection(v['seen'])]
        ranked.sort(key=lambda v: (len(ids.intersection(v['seen'])), v['time']), reverse=True)
        return copy.deepcopy(ranked[:limit])

    def observation_groups(self):
        indexed = {v['id']: v for v in self.views}
        result = []
        directions = ('front', 'right', 'back', 'left')
        for group in self.groups:
            frames, covered = [], set()
            for key in group['frame_ids']:
                frame = indexed[key]
                offset = angle_delta(frame['yaw'], group['yaw'])
                slot = round(offset/(math.pi/2)) % 4
                direction = directions[slot] if abs(angle_delta(offset, slot*math.pi/2)) < math.radians(20) else 'oblique'
                if direction != 'oblique':
                    covered.add(direction)
                frames.append(dict(frame_id=key, direction=direction, yaw_offset=offset,
                                   visible_objects=list(frame['seen'])))
            result.append(dict(view_id=group['id'], kind=group['kind'],
                               capture_pose=dict(position=group['position'], yaw=group['yaw']),
                               frames=frames, observed_directions=sorted(covered),
                               complete=group['kind']=='scan' and len(covered)==4))
        return copy.deepcopy(result)

    def context(self, ids, exclude=(), image_limit=3):
        """Derived summary, not another independently maintained memory store."""
        relevant = set(ids)
        groups = self.observation_groups()
        scored = sorted(groups, key=lambda g: (
            len(relevant.intersection(k for f in g['frames'] for k in f['visible_objects'])),
            g['view_id']), reverse=True)
        chosen = next((g for g in scored if any(relevant.intersection(f['visible_objects']) for f in g['frames'])), None)
        if chosen is None and groups:
            chosen = groups[-1]
        frames = []
        if chosen:
            by_id = {v['id']: v for v in self.views}
            candidates = [by_id[f['frame_id']] for f in chosen['frames'] if f['frame_id'] not in exclude]
            candidates.sort(key=lambda v: (len(relevant.intersection(v['seen'])), v['time']), reverse=True)
            frames = copy.deepcopy(candidates[:image_limit])
        history = [dict(view_id=g['view_id'], kind=g['kind'], position=g['capture_pose']['position'],
                        observed_directions=g['observed_directions'], complete=g['complete'],
                        source_frames=[f['frame_id'] for f in g['frames']]) for g in groups[-4:]]
        evidence = [dict(object_id=key, source_view=v['id'], observation_group=v['group_id'],
                         claim='visually_detected; category remains a hypothesis')
                    for v in frames for key in v['seen'] if key in relevant][:12]
        return dict(history=history, evidence=evidence,
                    selected_group=chosen, frames=frames,
                    meaning='Observed directions, not proof of semantic absence; cross-frame co-occurrence is not a relation')

    def summary(self):
        return copy.deepcopy(self.views)
