"""Program-owned single-object navigation; the model only inspects evidence."""
import base64
import io
import json
from pathlib import Path

from PIL import Image
from .annotations import draw_labels

from .geometry import horizontal_distance
from .task_context import excluded_ids
from .policy import DecisionValidationError


def evidence_area(evidence):
    box = evidence.get('bbox')
    return max(0., box[2]-box[0])*max(0., box[3]-box[1]) if box else 0.


def best_evidence(item):
    return max(item['evidence'], key=evidence_area)


def defer_candidate(policy, memory, key):
    deferred = set(getattr(policy, 'deferred_candidate_ids', [])) | {key}
    policy.deferred_candidate_ids = sorted(deferred)
    if not hasattr(policy, 'deferred_evidence'):
        policy.deferred_evidence = {}
    policy.deferred_evidence[key] = dict(memory.instances[key]['evidence'][-1])


def recover_candidate(policy, memory, key):
    """New information can reopen inspection; repeated identical views cannot."""
    if key not in getattr(policy, 'deferred_candidate_ids', []):
        return True
    old = getattr(policy, 'deferred_evidence', {}).get(key)
    fresh = memory.instances[key]['evidence'][-1]
    if not old or old['observation_id'] == fresh['observation_id']:
        return False
    improved = (evidence_area(fresh) > max(1., evidence_area(old))*1.5 or
                horizontal_distance(old['camera_position'], fresh['camera_position']) >= .6)
    counts = getattr(policy, 'reactivation_counts', {})
    if not improved or counts.get(key, 0) >= 2:
        return False
    counts[key] = counts.get(key, 0)+1
    policy.reactivation_counts = counts
    policy.deferred_candidate_ids = [k for k in policy.deferred_candidate_ids if k != key]
    policy.inspection_attempts[key] = 0
    policy.visual_checks.pop(key, None)
    return True


def evidence_content(observation, item, task, memory):
    key = item['id']
    category = item['category']
    evidence = item['evidence'][-1]
    visible = [i for i in memory.instances.values() if i['last_seen'] == observation['id']
               and i['category'] in task['target_categories']]
    excluded = excluded_ids(task, memory.instances)
    context = dict(observation_id=observation['id'], candidate_id=key,
                   detected_category=category,
                   target_categories=task['target_categories'], attributes=task.get('attributes', []),
                   identity_state=item.get('identity_state', 'unknown'),
                   visible_candidates=[dict(id=i['id'], detected_category=i['category'],
                                            bbox=i['evidence'][-1].get('bbox'), excluded=i['id'] in excluded)
                                       for i in visible],
                   excluded_previous_id=(task.get('excluded_goal') or {}).get('id'))
    content = [{'type': 'text', 'text': json.dumps(context, ensure_ascii=False)}]

    def add(image, label):
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='JPEG', quality=90)
        content.extend([{'type': 'text', 'text': label},
                        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' +
                         base64.b64encode(buffer.getvalue()).decode('ascii')}}])

    with Image.open(observation['rgb_path']) as source:
        scene = source.convert('RGB')
    draw_labels(scene, [dict(id=i['id'], category=i['category'], bbox=i['evidence'][-1].get('bbox'))
                        for i in visible], selected=key, excluded=excluded)
    scene.save(Path(observation['rgb_path']).with_name(observation['id']+'_'+key+'_annotated.jpg'), quality=90)
    add(scene, 'Current scene. Yellow box identifies candidate '+key+
        '; detected category: '+category+' (detector hypothesis, verify visually).')
    with Image.open(evidence['crop']) as crop:
        add(crop, 'Current candidate '+key+' from '+evidence['observation_id'])
    historical = best_evidence(item)
    if historical['observation_id'] != evidence['observation_id'] and evidence_area(historical) > evidence_area(evidence)*1.5:
        with Image.open(historical['crop']) as crop:
            add(crop, 'Historical clearer view of '+key+' from '+historical['observation_id']+
                '; not current arrival evidence.')
    anchor = (task.get('excluded_goal') or {}).get('id')
    previous = memory.instances.get(anchor)
    if previous and previous['evidence']:
        with Image.open(best_evidence(previous)['crop']) as crop:
            add(crop, 'Historical previous target '+anchor+'. Similar appearance alone does not establish identity.')
    return content


def inspect_candidate(policy, observation, item, memory):
    content = evidence_content(observation, item, policy.task, memory)
    policy.last_input = dict(stage='visual_check', image_policy='task_candidate',
                             candidate_ids=[item['id']], scene_view_ids=[],
                             image_count=sum(x['type']=='image_url' for x in content))
    answer = policy._request(
        '核实黄色框候选，同时留意全图中更符合任务的物体。青色框为其他候选，框上ID与visible_candidates对应。类别标签不是事实。'
        '判断目标类别、主体颜色/属性是否符合；遮挡不等于类别错误，可以结合历史清晰图理解物体，但不能用历史图替代当前到达确认。'
        '光照、遮挡或画面不清时回答unknown，不猜测。无属性要求时attributes_match=yes。'
        '返回JSON：{"candidate_id":"指定ID","observation_id":"当前观测ID",'
        '"category_match":"yes|no|unknown","attributes_match":"yes|no|unknown",'
        '"reason":"可见依据", "approach_recommended":false, "target_hint":null}。'
        '如果能明确辨认类别或目标属性，值得靠近核实另一项，approach_recommended可为true，并说明具体依据；'
        '仅有模糊暗块、类别和属性都unknown时不要建议直接靠近；明确不符时不能建议靠近。'
        '如果别的框更明确符合目标，target_hint返回{"candidate_id":"该框ID",'
        '"category_match":"yes","attributes_match":"yes","reason":"具体位置和可见依据"}。'
        '不要只在reason里提到其他目标，要填写target_hint。只推荐当前可见且未excluded的ID，不得编造ID。', content)
    if (not isinstance(answer, dict) or answer.get('candidate_id') != item['id'] or
            answer.get('observation_id') != observation['id'] or
            not isinstance(answer.get('reason'), str) or
            any(answer.get(field) not in ('yes', 'no', 'unknown')
                for field in ('category_match', 'attributes_match'))):
        raise DecisionValidationError('Visual answer must identify current candidate and evidence', answer)
    return answer


def grounded_hint(answer, observation, memory, task):
    hint = answer.get('target_hint')
    if not isinstance(hint, dict):
        return None
    key = hint.get('candidate_id')
    if not isinstance(key, str) or key not in memory.instances:
        return None
    item = memory.instances[key]
    if (item['last_seen'] != observation['id'] or key in excluded_ids(task, memory.instances) or
            item['category'] not in task['target_categories'] or
            hint.get('category_match') != 'yes' or hint.get('attributes_match') != 'yes' or
            not isinstance(hint.get('reason'), str) or not hint['reason'].strip()):
        return None
    return hint


def decide_verified(policy, observation, memory, selected, at_goal):
    task = policy.task
    identity_excluded = excluded_ids(task, memory.instances)
    seen = set(memory.observations[-1]['seen'])
    checks = policy.visual_checks
    if not hasattr(policy, 'inspection_attempts'):
        policy.inspection_attempts = {}
    for key in seen - identity_excluded:
        recover_candidate(policy, memory, key)
    excluded = identity_excluded | set(getattr(policy, 'deferred_candidate_ids', []))
    policy.last_input = dict(stage='program', candidate_ids=[], scene_view_ids=[], image_count=0)

    def action(kind, key=None, reason='Program navigation transition', **extra):
        return dict(action=kind, candidate_id=key, evidence_id=observation['id'], reason=reason, **extra)

    if selected in excluded:
        return action('explore', reason='Selected identity is excluded')
    if selected and selected not in seen:
        policy.inspection_attempts[selected] = policy.inspection_attempts.get(selected, 0)+1
        if policy.inspection_attempts[selected] >= 2:
            return action('defer', selected, 'Two new views did not reacquire target; continue search')
        return action('observe', selected, 'Target occluded; acquire a different view')
    # A positive initial visual check authorizes approach. Arrival always needs
    # a new current-frame check; no model calls are spent on repeated movement.
    if selected and not at_goal and (checks.get(selected, {}).get('matches') or checks.get(selected, {}).get('approach')):
        return action('select', selected, 'Continue approaching visually checked candidate')

    def worth_checking(item):
        old = checks.get(item['id'])
        if not old or old['matches'] or item['id'] == selected:
            return True
        return horizontal_distance(old['position'], observation['position']) >= .6

    def rank(item):
        # Weak color is useful for ordering questions, never for accepting a goal.
        colors = item.get('attributes', {}).get('color_scores', {})
        appearance = sum(colors.get(a, 0.) for a in task.get('attributes', []))
        return (bool(item.get('possible_same_ids')), -appearance,
                -item['confidence'], horizontal_distance(item['position'], observation['position']))

    candidates = [memory.instances[selected]] if selected else sorted(
        (i for i in memory.instances.values() if i['id'] in seen and i['id'] not in excluded
         and i['category'] in task['target_categories'] and worth_checking(i)),
        key=rank)[:2]
    uncertain = None
    for item in candidates:
        key = item['id']
        answer = inspect_candidate(policy, observation, item, memory)
        matches = answer['category_match'] == answer['attributes_match'] == 'yes'
        checks[key] = dict(matches=matches, observation_id=observation['id'],
                           position=list(observation['position']), answer=answer)
        hint = grounded_hint(answer, observation, memory, task)
        if hint and hint['candidate_id'] != key and not (matches and selected == key):
            suggested = hint['candidate_id']
            if recover_candidate(policy, memory, suggested):
                checks[suggested] = dict(matches=False, approach=True, observation_id=observation['id'],
                                          position=list(observation['position']), answer=hint)
                policy.inspection_attempts[suggested] = 0
                return action('select', suggested, hint['reason'], switch_reason=hint['reason'],
                              visual_hint=dict(hint, observation_id=observation['id']))
        if matches:
            policy.inspection_attempts[key] = 0
            if at_goal and selected == key:
                if item.get('views', 0) < 2:
                    return action('observe', key, 'Identity needs a second observation')
                return action('confirm', key, answer['reason'])
            return action('select', key, answer['reason'])
        mismatch = 'no' in (answer['category_match'], answer['attributes_match'])
        if (not at_goal and not mismatch and answer.get('approach_recommended') is True and
                'yes' in (answer['category_match'], answer['attributes_match']) and answer['reason'].strip()):
            checks[key]['approach'] = True
            recommendation = action('select', key, answer['reason'])
            if selected:
                return recommendation
            uncertain = recommendation
            continue
        if selected:
            if mismatch and not at_goal:
                return action('defer', key, answer['reason'])
            if mismatch and at_goal:
                return action('reject', key, answer['reason'], rejection_kind=(
                    'category_mismatch' if answer['category_match']=='no' else 'attribute_mismatch'))
            policy.inspection_attempts[key] = policy.inspection_attempts.get(key, 0)+1
            return action('defer' if policy.inspection_attempts[key] >= 2 else 'observe', key, answer['reason'])
        if not mismatch:
            uncertain = uncertain or action('observe', key, answer['reason'])
    if uncertain:
        return uncertain
    return action('explore', reason='No current visually supported candidate; continue coverage search')
