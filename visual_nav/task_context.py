"""Bind explicit 'another object' requests to a fixed, visually verified session goal."""
import copy
import re
import numpy as np


def requests_another(instruction):
    return bool(re.search(r'(?:另(?:外)?一|换一)(?:个|把|张|台|盏|只|件|辆)|\banother\b', instruction, re.I))


def goal_snapshot(instance):
    return copy.deepcopy({key: instance[key] for key in
                          ('id', 'category', 'position', 'bbox_min', 'bbox_max', 'last_seen')})


def bind_task(task, instruction, previous_goal):
    task = copy.deepcopy(task)
    if requests_another(instruction) or task.get('another'):
        if not previous_goal or previous_goal['category'] not in task['target_categories']:
            raise ValueError('“另一个”缺少已确认的同类目标作为参照，请先到达一个明确目标')
        task['another'] = True
        task['excluded_goal'] = copy.deepcopy(previous_goal)
    return task


def excluded_ids(task, instances):
    anchor = (task or {}).get('excluded_goal')
    if not anchor:
        return set()
    excluded = {anchor['id']}
    alo, ahi = np.asarray(anchor['bbox_min']), np.asarray(anchor['bbox_max'])
    for key, item in instances.items():
        if item['category'] != anchor['category']:
            continue
        if np.linalg.norm((np.asarray(item['position'])-anchor['position'])[[0,2]]) > .45:
            continue
        lo, hi = np.asarray(item['bbox_min']), np.asarray(item['bbox_max'])
        overlap = np.maximum(0, np.minimum(hi, ahi)-np.maximum(lo, alo))
        volume = min(np.prod(hi-lo), np.prod(ahi-alo))
        # Seat/back masks can describe different surfaces of the same chair and
        # have little volume overlap. Treat close touching fragments as an
        # ambiguous old identity, not proof of a different object.
        surface_gap = np.linalg.norm(np.maximum(0, np.maximum(lo-ahi, alo-hi)))
        if (volume > 1e-9 and np.prod(overlap)/volume >= .5) or surface_gap <= .15:
            excluded.add(key)
    return excluded
