"""Incremental visual instances, per-observation evidence, and revisable hypotheses."""
import copy
import numpy as np
from .geometry import horizontal_distance
from .relations import RelationMemory
from .views import ViewMemory


class ObjectMemory:
    def __init__(self, merge_distance=0.45):
        self.instances = {}
        self.merge_distance = merge_distance
        self.observations = []
        self.rejections = []
        self.questions = []
        self.relations = RelationMemory()
        self.views = ViewMemory()

    def integrate(self, detections, observation):
        seen, fresh = [], {}
        # Match the whole frame before updating any instance. Confidence order must
        # not let an ambiguous detection steal the best match of another object.
        prepared, pairs = [], []
        for index, detection in enumerate(detections):
            points = np.asarray(detection['points'])
            center = np.median(points, axis=0)
            lo, hi = np.quantile(points, [0.05, 0.95], axis=0)
            prepared.append((detection, points, center, lo, hi))
            for key, old in self.instances.items():
                if old['category'] != detection['category']:
                    continue
                distance = horizontal_distance(center, old['position'])
                overlap = np.minimum(hi, old['bbox_max'])-np.maximum(lo, old['bbox_min'])
                if distance < self.merge_distance and np.all(overlap > -.12):
                    pairs.append((distance, index, key))
        assignment, used = {}, set()
        for _, index, key in sorted(pairs):
            if index not in assignment and key not in used:
                assignment[index] = key
                used.add(key)
        for index, (detection, points, center, lo, hi) in enumerate(prepared):
            key = assignment.get(index) or 'v%04d' % (len(self.instances)+1)
            evidence = {k: detection[k] for k in ('crop', 'mask', 'confidence', 'attributes')}
            evidence.update(observation_id=observation['id'], time=observation['time'],
                            camera_position=observation['camera_to_world'][:3, 3].tolist())
            if key not in self.instances:
                self.instances[key] = dict(id=key, category=detection['category'],
                                          position=center.tolist(), bbox_min=lo.tolist(), bbox_max=hi.tolist(),
                                          confidence=detection['confidence'], evidence=[], attributes={}, views=0)
            item = self.instances[key]
            # Rolling robust surface samples preserve multiple viewpoints, but allow correction.
            old_points = np.asarray(item.get('points', []))
            fused = np.concatenate((old_points[-1500:], points)) if old_points.size else points
            item['points'] = fused[::max(1, len(fused)//2000)].tolist()
            item['position'] = np.median(fused, axis=0).tolist()
            item['bbox_min'], item['bbox_max'] = [a.tolist() for a in np.quantile(fused, [0.05, 0.95], axis=0)]
            item['confidence'] = float(0.5*item['confidence'] + 0.5*detection['confidence'])
            item['evidence'] = (item['evidence'] + [evidence])[-12:]
            # Recent evidence outweighs stale views; colors remain hypotheses, not labels.
            votes = {}
            for age, e in enumerate(reversed(item['evidence'])):
                for color, score in e['attributes'].get('color_scores', {}).items():
                    votes[color] = votes.get(color, 0.0) + (0.75**age)*float(score)
            total = sum(votes.values()) or 1
            item['attributes'] = {'color_scores': {k: v/total for k, v in votes.items()}}
            item['views'] += 1
            item['last_seen'] = observation['id']
            used.add(key)
            seen.append(key)
            fresh[key] = dict(bbox_min=lo.tolist(), bbox_max=hi.tolist(), confidence=float(detection['confidence']))
        self.observations.append({'id': observation['id'], 'rgb': observation['rgb_path'],
                                  'seen': seen, 'time': observation['time'],
                                  'meaning': 'visible detections only; absence is not evidence of an empty region'})
        self.relations.update(fresh, observation)
        self.views.add(observation, seen)
        return seen

    def reject(self, key, observation_id, reason):
        if key not in self.instances or self.instances[key]['last_seen'] != observation_id:
            raise ValueError('Rejection requires a current visual observation of this instance')
        self.rejections.append(dict(id=key, observation_id=observation_id, reason=reason))

    def summary(self):
        return {'instances': [{k: copy.deepcopy(v) for k, v in item.items() if k != 'points'}
                              for item in self.instances.values()],
                'rejections': copy.deepcopy(self.rejections), 'unresolved': list(self.questions),
                'relations': self.relations.summary(), 'scene_views': self.views.summary()}
