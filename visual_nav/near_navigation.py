"""Program-owned near-object tasks, reusing existing visual geometry and motion."""
import base64
import io
import json

from PIL import Image

from .annotations import draw_labels
from .geometry import horizontal_distance
from .inspection import reference_for
from .policy import DecisionValidationError
from .relations import pair_relations
from .task_context import excluded_ids


def relation_evidence(memory, target, reference, observation, final=False):
    """Current joint geometry or recent, genuinely different supporting views."""
    if target == reference or target not in memory.instances or reference not in memory.instances:
        return None
    if 'near' not in pair_relations(memory.instances[target], memory.instances[reference]):
        return None
    edge = next((e for e in memory.relations.summary() if e['relation'] == 'near'
                 and {e['source'], e['target']} == {target, reference}), None)
    if not edge or edge['status'] == 'contradicted':
        return None
    if edge['last_checked'] == observation['id']:
        return edge
    recent = [e for e in edge['evidence'] if e['supports']
              and 0 <= observation['time']-e['time'] <= 120]
    if not recent:
        return None
    if not final:
        return edge
    if edge['status'] == 'supported' and any(
            horizontal_distance(a['camera_position'], b['camera_position']) >= .3
            for a in recent for b in recent):
        return edge
    return None


def inspect_pair(policy, observation, memory, target, reference, edge):
    visible = [i for i in memory.instances.values() if i['last_seen'] == observation['id']
               and i['category'] in policy.task['target_categories']+policy.task['reference_categories']]
    context = dict(observation_id=observation['id'], candidate_id=target['id'],
                   reference_id=reference['id'], target_categories=policy.task['target_categories'],
                   reference_categories=policy.task['reference_categories'], attributes=policy.task.get('attributes', []),
                   relation='near', reference_current=reference['last_seen'] == observation['id'],
                   geometry=pair_relations(target, reference).get('near'),
                   relation_evidence_id=edge['last_checked'] if edge else None,
                   visible_candidates=[dict(id=i['id'], category=i['category'], bbox=i['evidence'][-1].get('bbox')) for i in visible])
    content = [dict(type='text', text=json.dumps(context, ensure_ascii=False))]
    def add(path, label, annotate=False):
        with Image.open(path) as source:
            image = source.convert('RGB')
        if annotate:
            draw_labels(image, context['visible_candidates'], selected=target['id'])
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        content.extend([dict(type='text', text=label), dict(type='image_url', image_url=dict(
            url='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()))])
    add(observation['rgb_path'], 'Current scene; yellow target, cyan other detected objects.', True)
    for item in (target, reference):
        evidence = item['evidence'][-1]
        add(evidence['crop'], item['id']+' crop from '+evidence['observation_id']+
            (' (historical, not current arrival evidence)' if evidence['observation_id'] != observation['id'] else ' (current)'))
    if reference['last_seen'] != observation['id'] and edge:
        evidence = next((e for e in reversed(edge['evidence']) if e['supports']), None)
        if evidence:
            add(evidence['rgb_path'], 'Historical joint scene '+evidence['observation_id']+
                ' containing '+target['id']+' and '+reference['id']+'; not current arrival evidence.')
    policy.last_input = dict(stage='relation_check', image_policy='near_pair',
        candidate_ids=[target['id'], reference['id']], image_count=sum(c['type']=='image_url' for c in content))
    answer = policy._request(
        '核实指定目标与参照物，不决定移动或搜索。标签是检测假设，不是真值。'
        '判断目标类别、目标属性、参照物类别、两者是否旁边；结合场景和几何，不能仅凭裁剪图判定关系。'
        '允许可靠的历史共同场景帮助判断关系，但历史图不能证明当前到达。无属性要求时attributes_match=yes。'
        '不清楚回答unknown，明确不符回答no。只返回JSON：'
        '{"candidate_id":"指定目标ID","reference_id":"指定参照ID","observation_id":"当前观测ID",'
        '"category_match":"yes|no|unknown","attributes_match":"yes|no|unknown",'
        '"reference_match":"yes|no|unknown","relation_match":"yes|no|unknown","reason":"具体视觉依据"}。', content)
    fields = ('category_match', 'attributes_match', 'reference_match', 'relation_match')
    if (not isinstance(answer, dict) or answer.get('candidate_id') != target['id']
            or answer.get('reference_id') != reference['id'] or answer.get('observation_id') != observation['id']
            or any(answer.get(f) not in ('yes', 'no', 'unknown') for f in fields)
            or not isinstance(answer.get('reason'), str) or not answer['reason'].strip()):
        raise DecisionValidationError('Relation answer must identify the pair and current observation', answer)
    return answer


def decide_near(policy, observation, memory, selected, at_goal):
    task = policy.task
    seen = set(memory.observations[-1]['seen'])
    excluded = excluded_ids(task, memory.instances) | set(getattr(policy, 'deferred_candidate_ids', []))
    if not hasattr(policy, 'near_checks'):
        policy.near_checks, policy.near_attempts = {}, {}
        policy.near_rejected_pairs = set()
    policy.last_input = dict(stage='program', image_count=0, candidate_ids=[])
    def action(kind, key=None, reason='', **extra):
        return dict(action=kind, candidate_id=key, evidence_id=observation['id'], reason=reason, **extra)
    if selected and selected not in excluded:
        candidates = [memory.instances[selected]]
    else:
        candidates = sorted((i for i in memory.instances.values() if i['id'] in seen-excluded
            and i['category'] in task['target_categories']), key=lambda i: -i['confidence'])[:2]
    fallback = None
    for target in candidates:
        key = target['id']
        reference = reference_for(key, task, memory)
        if key not in task.get('near_references', {}) and reference and (key, reference['id']) in policy.near_rejected_pairs:
            alternatives = [i for i in memory.instances.values() if i['id'] != key
                and i['category'] in task['reference_categories']
                and (key, i['id']) not in policy.near_rejected_pairs]
            reference = min(alternatives, key=lambda i: horizontal_distance(i['position'], target['position']), default=None)
        if reference is None:
            continue
        edge = relation_evidence(memory, key, reference['id'], observation, final=at_goal)
        cached = policy.near_checks.get((key, reference['id']))
        if key not in seen or (cached and cached['matches'] and not edge):
            policy.near_attempts[key] = policy.near_attempts.get(key, 0)+1
            return action('defer' if policy.near_attempts[key] >= 2 else 'observe', key,
                          'Acquire missing current target or reliable joint relation evidence')
        if cached and cached['matches'] and not at_goal:
            return action('select', key, 'Continue approaching verified target/reference pair')
        if cached and not cached['matches'] and key != selected and horizontal_distance(
                cached['position'], observation['position']) < .6:
            continue
        answer = inspect_pair(policy, observation, memory, target, reference, edge)
        matches = all(answer[f] == 'yes' for f in ('category_match','attributes_match','reference_match','relation_match')) and edge is not None
        policy.near_checks[(key, reference['id'])] = dict(matches=matches, position=list(observation['position']))
        if matches:
            task.setdefault('near_references', {})[key] = reference['id']
            policy.near_attempts[key] = 0
            if at_goal and key == selected:
                return action('confirm', key, answer['reason'], reference_id=reference['id'],
                    reference_verified=True, reference_reason=answer['reason'])
            return action('select', key, answer['reason'])
        if selected:
            policy.near_attempts[key] = policy.near_attempts.get(key, 0)+1
            mismatch = any(answer[f] == 'no' for f in ('category_match','attributes_match','reference_match','relation_match'))
            return action('defer' if mismatch or policy.near_attempts[key] >= 2 else 'observe', key, answer['reason'])
        if answer['reference_match'] == 'no' or answer['relation_match'] == 'no':
            policy.near_rejected_pairs.add((key, reference['id']))
        if not any(answer[f] == 'no' for f in ('category_match','attributes_match','reference_match','relation_match')):
            task.setdefault('near_references', {})[key] = reference['id']
            fallback = action('observe', key, 'Need joint evidence for target/reference pair')
    return fallback or action('observe' if selected else 'explore', selected, 'Search for target/reference evidence')
