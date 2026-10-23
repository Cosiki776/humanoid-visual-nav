#!/usr/bin/env python3
"""OFFLINE GT evaluator. Never import this module from a visual policy or executor."""
import argparse
import json
from pathlib import Path
import numpy as np


def associate(points, objects, return_all=False):
    """Assign an estimated surface cloud to a concrete GT instance, or abstain.

    AABB overlap is a transparent first-version metric, not perfect mesh identity.
    Ambiguous overlapping boxes are unresolved and cannot count as success.
    """
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 25 or not np.isfinite(p).all():
        return None, []
    ranking = []
    for obj in objects:
        lo, hi = np.asarray(obj['bbox_min'])-.08, np.asarray(obj['bbox_max'])+.08
        fraction = float(np.all((p >= lo) & (p <= hi), axis=1).mean())
        ranking.append((fraction, obj['id']))
    ranking.sort(reverse=True)
    if not ranking or ranking[0][0] < .5 or (len(ranking) > 1 and ranking[0][0]-ranking[1][0] < .15):
        return None, ranking if return_all else ranking[:3]
    return ranking[0][1], ranking if return_all else ranking[:3]


def score_episode(result, episode, objects):
    accepted = episode['goal_ids']
    indexed = {o['id']: o for o in objects}
    if any(k not in indexed for k in accepted):
        raise ValueError('Episode references unknown GT instance')
    predicted = result.get('target_object') if result.get('mode') == 'gt' else None
    ranking = []
    if result.get('mode') != 'gt':
        predicted, ranking = associate(result.get('selected_points', []), objects, return_all=True)
    claimed = result['status'] in {'succeeded', 'arrived_unverified'}
    gap = None
    if predicted in indexed:
        obj = indexed[predicted]
        p = np.asarray(result['final_position'])[[0, 2]]
        lo, hi = np.asarray(obj['bbox_min'])[[0, 2]], np.asarray(obj['bbox_max'])[[0, 2]]
        gap = float(np.linalg.norm(np.maximum(np.maximum(lo-p, p-hi), 0)))
    correct = predicted in accepted
    plausible = {key for fraction, key in ranking if fraction >= .5}
    definitely_wrong = claimed and ((predicted is not None and not correct) or
                                   (bool(plausible) and plausible.isdisjoint(accepted)))
    min_gap, max_gap = episode.get('min_target_gap_m', .2), episode.get('max_target_gap_m', 1.2)
    success = bool(accepted and claimed and correct and gap is not None and min_gap <= gap <= max_gap)
    return dict(episode=episode['id'], mode=result['mode'], policy=result.get('policy'),
                goal_present=bool(accepted), success=success,
                wrong_target=bool(definitely_wrong),
                unresolved_identity=bool(claimed and predicted is None),
                abstained=not claimed, matched_instance=predicted, association=ranking[:3],
                target_gap_m=gap, verified=result.get('verified', False), status=result['status'],
                path_length_m=result['path_length_m'], observations=result['observations'],
                elapsed_seconds=result['elapsed_seconds'])


def aggregate(rows):
    groups = {}
    for row in rows:
        key = row['mode']+'/'+str(row.get('policy'))
        groups.setdefault(key, []).append(row)
    out = {}
    for key, group in groups.items():
        positives = [r for r in group if r['goal_present']]
        negatives = [r for r in group if not r['goal_present']]
        out[key] = dict(n=len(group), positive_n=len(positives), negative_n=len(negatives),
                        success_rate=sum(r['success'] for r in positives)/len(positives) if positives else None,
                        wrong_target_rate=sum(r['wrong_target'] for r in group)/len(group),
                        error_rate=sum(r.get('status') == 'error' for r in group)/len(group),
                        unresolved_identity_rate=sum(r['unresolved_identity'] for r in group)/len(group),
                        negative_abstention_rate=sum(r['abstained'] and r.get('status') != 'error'
                                                     for r in negatives)/len(negatives) if negatives else None,
                        mean_path_length_m=float(np.mean([r['path_length_m'] for r in group])),
                        mean_observations=float(np.mean([r['observations'] for r in group])),
                        mean_elapsed_seconds=float(np.mean([r['elapsed_seconds'] for r in group])))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export-scene')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gt', type=Path)
    p.add_argument('--episode', type=Path)
    p.add_argument('--result', type=Path)
    args = p.parse_args()
    if args.export_scene:
        import bootstrap
        from world import World
        world = World(args.export_scene)
        try:
            data = dict(scene=args.export_scene, objects=world.objects())
        finally:
            world.close()
    else:
        if not all((args.gt, args.episode, args.result)):
            p.error('Scoring requires --gt --episode --result')
        gt = json.loads(args.gt.read_text())
        episode = json.loads(args.episode.read_text())
        config_path = args.result.parent/'config.json'
        config = json.loads(config_path.read_text())
        if episode['scene'] != gt['scene'] or config['scene'] != gt['scene']:
            raise ValueError('Scene mismatch')
        data = score_episode(json.loads(args.result.read_text()), episode, gt['objects'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(data if not args.export_scene else {'exported': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
