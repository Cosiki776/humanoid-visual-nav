"""Incremental visual instances, per-observation evidence, and revisable hypotheses."""
import copy
import numpy as np
from .geometry import horizontal_distance
from .relations import RelationMemory
from .views import ViewMemory
from .ledger import TaskLedger


class ObjectMemory:
    def __init__(self, merge_distance=0.45):
        self.instances = {}
        self.merge_distance = merge_distance
        self.observations = []
        self.rejections = []
        self.questions = []
        self.relations = RelationMemory()
        self.views = ViewMemory()
        self.ledger = TaskLedger()
        self.version = 0
        self.aliases = {}
        self.next_id = 1

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
                distance = horizontal_distance(center, old['position'])
                overlap = np.minimum(hi, old['bbox_max'])-np.maximum(lo, old['bbox_min'])
                # Cross-category identity needs strong 3-D overlap, not proximity alone.
                volume = min(np.prod(hi-lo), np.prod(np.asarray(old['bbox_max'])-old['bbox_min']))
                overlap_ratio = np.prod(np.maximum(overlap, 0))/max(float(volume), 1e-9)
                same_category = old['category'] == detection['category']
                if (same_category or (distance < .15 and overlap_ratio > .75)) and distance < self.merge_distance and np.all(overlap > -.12):
                    pairs.append((distance, index, key))
        assignment, used = {}, set()
        for _, index, key in sorted(pairs):
            if index not in assignment and key not in used:
                assignment[index] = key
                used.add(key)
        for index, (detection, points, center, lo, hi) in enumerate(prepared):
            key = assignment.get(index)
            if key is None:
                key = 'v%04d' % self.next_id
                self.next_id += 1
            evidence = {k: detection[k] for k in ('crop', 'mask', 'confidence', 'attributes')}
            if 'bbox_2d' in detection:
                evidence['bbox_2d'] = list(detection['bbox_2d'])
            evidence.update(observation_id=observation['id'], time=observation['time'],
                            camera_position=observation['camera_to_world'][:3, 3].tolist())
            if key not in self.instances:
                self.instances[key] = dict(id=key, category=detection['category'],
                                          position=center.tolist(), bbox_min=lo.tolist(), bbox_max=hi.tolist(),
                                          confidence=detection['confidence'], evidence=[], attributes={}, views=0)
            item = self.instances[key]
            beliefs = item.setdefault('category_scores', {})
            for category in beliefs:
                beliefs[category] *= .75
            beliefs[detection['category']] = beliefs.get(detection['category'], 0) + float(detection['confidence'])
            item['category'] = max(beliefs, key=beliefs.get)
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
        # Repair conservative duplicate identities only if never seen as two objects together.
        for key in list(seen):
            if key not in self.instances:
                continue
            item = self.instances[key]
            matches = []
            for other, old in self.instances.items():
                if other >= key or old['category'] != item['category']:
                    continue
                if horizontal_distance(old['position'], item['position']) >= .18:
                    continue
                overlap = np.maximum(0, np.minimum(old['bbox_max'], item['bbox_max'])-np.maximum(old['bbox_min'], item['bbox_min']))
                v1 = np.prod(np.asarray(old['bbox_max'])-old['bbox_min'])
                v2 = np.prod(np.asarray(item['bbox_max'])-item['bbox_min'])
                iou = np.prod(overlap)/max(v1+v2-np.prod(overlap), 1e-9)
                if iou > .8 and not any(key in o['seen'] and other in o['seen'] for o in self.observations):
                    matches.append(other)
            if len(matches) == 1:
                self.merge_alias(key, matches[0])
        seen = list(dict.fromkeys(self.resolve(key) for key in seen))
        self.version += 1
        return seen

    def resolve(self, key):
        while key in self.aliases:
            key = self.aliases[key]
        return key

    def merge_alias(self, alias, canonical):
        """Caller must establish geometric identity; migrate every evidence reference."""
        alias, canonical = self.resolve(alias), self.resolve(canonical)
        if alias == canonical:
            return
        if any(alias in o['seen'] and canonical in o['seen'] for o in self.observations):
            raise ValueError('Cannot merge independently co-observed objects')
        old, keep = self.instances[alias], self.instances[canonical]
        evidence = {e['observation_id']: e for e in keep['evidence']+old['evidence']}
        keep['evidence'] = sorted(evidence.values(), key=lambda e: e['time'])[-12:]
        keep['views'] += old['views']
        keep['last_seen'] = keep['evidence'][-1]['observation_id']
        self.aliases[alias] = canonical
        del self.instances[alias]
        for observation in self.observations:
            observation['seen'] = list(dict.fromkeys(self.resolve(k) for k in observation['seen']))
        for view in self.views.views:
            view['seen'] = list(dict.fromkeys(self.resolve(k) for k in view['seen']))
        for rejection in self.rejections:
            rejection['id'] = self.resolve(rejection['id'])
        self.ledger.remap(alias, canonical)
        edges = {}
        for edge in self.relations.summary():
            edge['source'], edge['target'] = self.resolve(edge['source']), self.resolve(edge['target'])
            if edge['source'] == edge['target']:
                continue
            if edge['relation'] == 'near':
                edge['source'], edge['target'] = sorted((edge['source'], edge['target']))
            key = (edge['source'], edge['relation'], edge['target'])
            if key not in edges or edge['updated_at'] >= edges[key]['updated_at']:
                edges[key] = edge
        self.relations.edges = edges

    def reject(self, key, observation_id, reason, kind=None, anchor=None):
        if key not in self.instances or self.instances[key]['last_seen'] != observation_id:
            raise ValueError('Rejection requires a current visual observation of this instance')
        claim = self.ledger.record(key, observation_id, reason, kind, anchor)
        self.rejections.append(dict(id=key, observation_id=observation_id, reason=reason, **{k: claim[k] for k in ('kind', 'task_id', 'status')}))

    def summary(self):
        return {'aliases': dict(self.aliases), 'version': self.version, 'task_ledger': self.ledger.summary(), 'instances': [{k: copy.deepcopy(v) for k, v in item.items() if k != 'points'}
                              for item in self.instances.values()],
                'rejections': copy.deepcopy(self.rejections), 'unresolved': list(self.questions),
                'relations': self.relations.summary(), 'scene_views': self.views.summary()}
