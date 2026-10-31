"""Choose joint-evidence viewpoints from estimated instances, never scene labels."""
import math
import numpy as np
from .geometry import horizontal_distance, path_length


def reference_for(target_id, task, memory):
    target = memory.instances[target_id]
    active = memory.ledger.active_anchor(memory, task or {})
    if active and active != target_id:
        return memory.instances[active]
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


def rank_reveal_viewpoints(points, current, anchor, route, visible_probe, visited=()):
    """Prefer new sides and newly visible space around an occluder, not its face.

    Probes are geometric hypotheses, never assumed target positions. No target
    detection or continued visibility of the anchor is required.
    """
    current = np.asarray(current)
    lo, hi = np.asarray(anchor['bbox_min']), np.asarray(anchor['bbox_max'])
    center = (lo+hi)/2
    probes = [np.array([center[0]+((hi[0]-lo[0])/2+.65)*math.cos(a),
                       current[1]+height,
                       center[2]+((hi[2]-lo[2])/2+.65)*math.sin(a)])
              for height in (.65, 1.2) for a in np.linspace(0, 2*math.pi, 16, endpoint=False)]
    def eye(point): return np.asarray(point)+[0,1.65,0]
    old_positions = [current]+[np.asarray(p) for p in visited]
    old_visible = [any(visible_probe(eye(p), probe) for p in old_positions) for probe in probes]
    def bearing(p): return math.atan2(p[0]-center[0], p[2]-center[2])
    ranked = []
    for point in points:
        point = np.asarray(point)
        if any(horizontal_distance(point, old) < .6 for old in old_positions):
            continue
        # A new coordinate on the same side does not uncover the hidden side.
        novelty = min(abs(math.atan2(math.sin(bearing(point)-bearing(old)),
                                     math.cos(bearing(point)-bearing(old)))) for old in old_positions)
        if novelty < math.radians(35):
            continue
        path = route(point)
        if path is None or not .5 <= path_length(path) <= 12.:
            continue
        visible = [visible_probe(eye(point), probe) for probe in probes]
        if not any(visible):
            continue
        gain = sum(new and not old for new, old in zip(visible, old_visible))
        ranked.append((gain, min(novelty, math.pi/2)/path_length(path), point))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [point for _, _, point in ranked]
