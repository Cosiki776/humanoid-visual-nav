"""Program-owned single-object navigation; the model only inspects evidence."""
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .geometry import horizontal_distance
from .task_context import excluded_ids
from .policy import DecisionValidationError


def evidence_content(observation, item, task, memory):
    key = item['id']
    category = item['category']
    evidence = item['evidence'][-1]
    context = dict(observation_id=observation['id'], candidate_id=key,
                   detected_category=category,
                   target_categories=task['target_categories'], attributes=task.get('attributes', []),
                   identity_state=item.get('identity_state', 'unknown'),
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
    draw = ImageDraw.Draw(scene)
    box = evidence.get('bbox')
    if box is not None:
        draw.rectangle(tuple(box), outline=(255, 210, 0), width=3)
        label = key+' '+category
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 16)
        except OSError:
            font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        width, height = right-left+8, bottom-top+6
        x = max(0, min(box[0], scene.width-width))
        y = max(0, box[1]-height)
        draw.rectangle((x, y, x+width, y+height), fill=(255, 210, 0))
        draw.text((x+4-left, y+3-top), label, font=font, fill=(0, 0, 0))
    scene.save(Path(observation['rgb_path']).with_name(observation['id']+'_'+key+'_annotated.jpg'), quality=90)
    add(scene, 'Current scene. Yellow box identifies candidate '+key+
        '; detected category: '+category+' (detector hypothesis, verify visually).')
    with Image.open(evidence['crop']) as crop:
        add(crop, 'Current candidate '+key+' from '+evidence['observation_id'])
    anchor = (task.get('excluded_goal') or {}).get('id')
    previous = memory.instances.get(anchor)
    if previous and previous['evidence']:
        with Image.open(previous['evidence'][-1]['crop']) as crop:
            add(crop, 'Historical previous target '+anchor+'. Similar appearance alone does not establish identity.')
    return content


def inspect_candidate(policy, observation, item, memory):
    content = evidence_content(observation, item, policy.task, memory)
    policy.last_input = dict(stage='visual_check', image_policy='task_candidate',
                             candidate_ids=[item['id']], scene_view_ids=[],
                             image_count=sum(x['type']=='image_url' for x in content))
    answer = policy._request(
        '只检查指定框中的物体，不决定机器人动作。类别标签不是事实。'
        '判断它是否是目标类别、主体颜色/属性是否符合；衣物、椅面局部或柜内物品不能当作完整目标类别的充分证据。'
        '光照、遮挡或画面不清时回答unknown，不猜测。无属性要求时attributes_match=yes。'
        '返回JSON：{"candidate_id":"指定ID","observation_id":"当前观测ID",'
        '"category_match":"yes|no|unknown","attributes_match":"yes|no|unknown",'
        '"reason":"可见依据"}。历史图只提供上下文，不能替代当前图。', content)
    if (not isinstance(answer, dict) or answer.get('candidate_id') != item['id'] or
            answer.get('observation_id') != observation['id'] or
            not isinstance(answer.get('reason'), str) or
            any(answer.get(field) not in ('yes', 'no', 'unknown')
                for field in ('category_match', 'attributes_match'))):
        raise DecisionValidationError('Visual answer must identify current candidate and evidence', answer)
    return answer


def decide_verified(policy, observation, memory, selected, at_goal):
    task = policy.task
    excluded = excluded_ids(task, memory.instances) | set(getattr(policy, 'deferred_candidate_ids', []))
    seen = set(memory.observations[-1]['seen'])
    checks = policy.visual_checks
    if not hasattr(policy, 'inspection_attempts'):
        policy.inspection_attempts = {}
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
    if selected and not at_goal and checks.get(selected, {}).get('matches'):
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
    for item in candidates:
        key = item['id']
        answer = inspect_candidate(policy, observation, item, memory)
        matches = answer['category_match'] == answer['attributes_match'] == 'yes'
        checks[key] = dict(matches=matches, observation_id=observation['id'],
                           position=list(observation['position']), answer=answer)
        if matches:
            policy.inspection_attempts[key] = 0
            if at_goal and selected == key:
                if item.get('views', 0) < 2:
                    return action('observe', key, 'Identity needs a second observation')
                return action('confirm', key, answer['reason'])
            return action('select', key, answer['reason'])
        mismatch = 'no' in (answer['category_match'], answer['attributes_match'])
        if selected:
            if mismatch and not at_goal:
                return action('defer', key, answer['reason'])
            if mismatch and at_goal:
                return action('reject', key, answer['reason'], rejection_kind=(
                    'category_mismatch' if answer['category_match']=='no' else 'attribute_mismatch'))
            policy.inspection_attempts[key] = policy.inspection_attempts.get(key, 0)+1
            return action('defer' if policy.inspection_attempts[key] >= 2 else 'observe', key, answer['reason'])
        if not mismatch:
            return action('observe', key, answer['reason'])
    return action('explore', reason='No current visually supported candidate; continue coverage search')
