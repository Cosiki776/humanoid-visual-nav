"""Choose joint-evidence viewpoints from estimated instances, never scene labels."""
import math
import numpy as np
from .geometry import horizontal_distance, path_length


def reference_for(target_id, task, memory):
    target = memory.instances[target_id]
    if target['category'] in (task or {}).get('reference_categories', []) and target['category'] not in (task or {}).get('target_categories', []):
        return None
    candidates = [i for i in memory.instances.values()
                  if i['category'] in (task or {}).get('reference_categories', []) and i['id'] != target_id
                  and not memory.ledger.blocked(target_id, task, i['id'])
                  and i['id'] not in memory.ledger.deferred
                  and memory.ledger.anchor_verdict(i['id'], i['category']) is not False]
    links = {edge['target'] if edge['source'] == target_id else edge['source']
             for edge in memory.relations.summary()
             if target_id in (edge['source'], edge['target']) and edge['status'] != 'contradicted'
             and edge['relation'] == (task or {}).get('relation')}
    return min(candidates, key=lambda i: (i['id'] not in links,
               horizontal_distance(i['position'], target['position'])), default=None)


def joint_aim(target, reference=None):
    center = np.asarray(target['position'])
    return center if reference is None else (center+np.asarray(reference['position']))/2


def rank_viewpoints(points, current, target, reference, route, visibility, visited=()):
    ranked = []
    for point in points:
        point = np.asarray(point)
        if horizontal_distance(point, current) < .5 or any(horizontal_distance(point, p) < .4 for p in visited):
            continue
        path = route(point)
        if path is None:
            continue
        eye = point+np.array([0, 1.65, 0])
        aim = joint_aim(target, reference)
        direction = aim-eye
        direction /= max(np.linalg.norm(direction), 1e-6)
        objects = [target] if reference is None else [target, reference]
        visible = 0
        for item in objects:
            ray = np.asarray(item['position'])-eye
            angular = np.dot(ray/max(np.linalg.norm(ray), 1e-6), direction)
            if angular >= math.cos(math.radians(35)) and visibility(eye, item):
                visible += 1
        if visible != len(objects):
            continue
        ranked.append((path_length(path), point))
    return [point for _, point in sorted(ranked, key=lambda x: x[0])]
