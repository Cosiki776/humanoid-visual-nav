"""Conservative, revisable geometric relations from co-observed visual instances.

These are hypotheses about visible surfaces, not physical contact or room labels.
Habitat world axes: Y up; distances in metres.
"""
import copy
from itertools import combinations
import numpy as np


def pair_relations(a, b):
    alo, ahi = np.asarray(a['bbox_min']), np.asarray(a['bbox_max'])
    blo, bhi = np.asarray(b['bbox_min']), np.asarray(b['bbox_max'])
    if not np.isfinite([alo, ahi, blo, bhi]).all():
        return {}
    overlap = np.maximum(0, np.minimum(ahi, bhi)-np.maximum(alo, blo))
    smaller_volume = min(np.prod(ahi-alo), np.prod(bhi-blo))
    # Different category hypotheses can describe the very same detected surface.
    if smaller_volume <= 1e-9 or np.prod(overlap)/smaller_volume > .5:
        return {}
    gap = np.maximum(0, np.maximum(alo-bhi, blo-ahi))
    horizontal_gap = float(np.linalg.norm(gap[[0, 2]]))
    center_distance = float(np.linalg.norm(((alo+ahi-blo-bhi)/2)[[0, 2]]))
    metrics = dict(horizontal_gap_m=horizontal_gap, center_distance_m=center_distance,
                   vertical_gap_m=float(gap[1]))
    result = {}
    if horizontal_gap <= 1.0 and center_distance <= 2.5 and gap[1] <= .5:
        result['near'] = metrics
    smaller_area = min(np.prod((ahi-alo)[[0, 2]]), np.prod((bhi-blo)[[0, 2]]))
    coverage = float(np.prod(overlap[[0, 2]])/max(smaller_area, 1e-9))
    if coverage >= .3:
        if .05 <= alo[1]-bhi[1] <= 1.0:
            result['a_above_b'] = dict(metrics, horizontal_overlap_ratio=coverage)
        if .05 <= blo[1]-ahi[1] <= 1.0:
            result['b_above_a'] = dict(metrics, horizontal_overlap_ratio=coverage)
    return result


class RelationMemory:
    def __init__(self):
        self.edges = {}

    def update(self, fresh, observation):
        """Only joint observation can support or contradict a pair; absence does neither."""
        proposals = {}
        for a, b in combinations(sorted(fresh), 2):
            for kind, metrics in pair_relations(fresh[a], fresh[b]).items():
                source, target, relation = (b, a, 'above') if kind == 'b_above_a' else (
                    a, b, 'above' if kind == 'a_above_b' else 'near')
                proposals[(source, relation, target)] = metrics
        keys = set(proposals)
        keys.update(key for key in self.edges if key[0] in fresh and key[2] in fresh)
        for key in sorted(keys):
            source, relation, target = key
            old = self.edges.get(key)
            if old and old['last_checked'] == observation['id']:
                continue
            supports = key in proposals
            edge = old or dict(source=source, relation=relation, target=target,
                               first_seen=observation['id'], supporting_observations=0, evidence=[])
            score = min(float(fresh[source]['confidence']), float(fresh[target]['confidence']))
            edge['supporting_observations'] = edge['supporting_observations']+1 if supports else 0
            edge['status'] = ('supported' if edge['supporting_observations'] >= 2 else 'hypothesis') if supports else 'contradicted'
            edge['confidence'] = round(min(.85, score*(1 if edge['status']=='supported' else .7)), 3) if supports else 0.0
            edge['last_checked'] = observation['id']
            edge['updated_at'] = observation['time']
            if supports:
                edge['last_supported'] = observation['id']
                edge['geometry'] = proposals[key]
            evidence = dict(observation_id=observation['id'], rgb_path=observation['rgb_path'],
                            time=observation['time'], supports=supports,
                            camera_position=np.asarray(observation['camera_to_world'])[:3, 3].tolist())
            edge['evidence'] = (edge['evidence']+[evidence])[-8:]
            self.edges[key] = edge

    def summary(self):
        return copy.deepcopy(list(self.edges.values()))
