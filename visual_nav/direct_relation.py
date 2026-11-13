"""Raw image -> VLM pair boxes -> SAM/depth -> existing geometric controller."""
import base64
import json
import math
from pathlib import Path

from PIL import Image

from .annotations import draw_labels
from .policy import DecisionValidationError
from .task_context import excluded_ids


PROMPT = (
    '根据用户任务直接查看原始全图，找到满足属性和旁边关系的目标及参照物。没有检测器候选，不要猜物体ID。'
    'target_attributes仅约束target，不约束reference；例如冰箱旁边的白色椅子，是椅子要白色，冰箱可以是任何颜色。'
    '如果附有已选目标的历史图及框，当前需要定位同一对物体，不是随意换另一个；历史图仅帮助辨认身份，返回框必须对应当前图。'
    '仅在两个物体确实可辨认、目标属性和关系均符合时found=true，否则found=false；不能把门、挂画当冰箱。'
    '定位整个物体的可见区域，不能只框把手、侧边或局部零件，框不要包含大片背景。'
    '注意室内物体在纹理和光照下的外观变化，根据结构与把手判断，不要仅凭颜色猜测类别。'
    '坐标为0到1000归一化的[x_min,y_min,x_max,y_max]，对应图像宽高。'
    '不判断是否已经走到、不输出运动动作或世界坐标。返回JSON：'
    '{"observation_id":"输入观测ID","found":true或false,"reason":"可见依据",'
    '"target":{"category":"任务目标类别","bbox":[0,0,1000,1000]},'
    '"reference":{"category":"任务参照类别","bbox":[0,0,1000,1000]}}。'
    'found=false时target和reference为null。')


def locate_pair(policy, obs, task):
    context = dict(observation_id=obs['id'], instruction=task.get('instruction'),
        target_categories=task['target_categories'], target_attributes=task.get('attributes', []),
        reference_categories=task['reference_categories'], reference_attributes=[], relation=task['relation'],
        coordinate_system='xyxy normalized 0..1000')
    content = [dict(type='text', text=json.dumps(context, ensure_ascii=False)),
        dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,'+
            base64.b64encode(Path(obs['rgb_path']).read_bytes()).decode()))]
    anchor = getattr(policy, 'direct_relation_anchor', None)
    if anchor:
        content.extend([dict(type='text',text='Historical selected pair for identity only: '+json.dumps(
            {k:anchor[k] for k in ('observation_id','target','reference')},ensure_ascii=False)),
            dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+
                 base64.b64encode(Path(anchor['rgb_path']).read_bytes()).decode()))])
    policy.last_input = dict(stage='direct_relation_grounding', image_count=2 if anchor else 1, candidate_ids=[])
    answer = policy._request(PROMPT, content)
    if (not isinstance(answer, dict) or answer.get('observation_id') != obs['id']
            or type(answer.get('found')) is not bool or not isinstance(answer.get('reason'), str)):
        raise DecisionValidationError('Grounding response must identify current frame and found flag', answer)
    if not answer['found']:
        return answer, []
    with Image.open(obs['rgb_path']) as source:
        width, height = source.size
    boxes = []
    for role, categories in [('target', task['target_categories']), ('reference', task['reference_categories'])]:
        item = answer.get(role)
        box = item.get('bbox') if isinstance(item, dict) else None
        if (not isinstance(item, dict) or item.get('category') not in categories
                or not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise DecisionValidationError('Invalid grounded '+role+' box/category', answer)
        pixel = [box[0]*width/1000, box[1]*height/1000, box[2]*width/1000, box[3]*height/1000]
        if pixel[2]-pixel[0] < 3 or pixel[3]-pixel[1] < 3:
            raise DecisionValidationError('Grounded box is too small for depth', answer)
        boxes.append(dict(role=role, category=item['category'], bbox=pixel))
    return answer, boxes


class DirectRelationPerception:
    def __init__(self, segmenter, policy, task):
        self.segmenter, self.policy, self.task = segmenter, policy, task
        self.frame = None

    def detect(self, obs, output, categories=()):
        try:
            answer, boxes = locate_pair(self.policy, obs, self.task)
        except DecisionValidationError as exc:
            answer, boxes = dict(found=False, reason='Invalid model localization: '+str(exc)), []
        detections = self.segmenter.segment_boxes(obs, output, boxes) if boxes else []
        self.frame = dict(observation_id=obs['id'], answer=answer, detections=detections)
        self.policy.direct_relation_frame = self.frame
        # Preview is derived from VLM output, never fed back as the raw input.
        with Image.open(obs['rgb_path']) as source:
            preview = source.convert('RGB')
        draw_labels(preview, [dict(id=b['role'], category=b['category'], bbox=b['bbox']) for b in boxes], selected='target')
        preview.save(Path(output)/(obs['id']+'_vlm_boxes.jpg'))
        return detections


def decide_direct(policy, obs, memory, selected, at_goal):
    def action(kind, key=None, reason='', **extra):
        return dict(action=kind, candidate_id=key, evidence_id=obs['id'], reason=reason, **extra)
    frame = getattr(policy, 'direct_relation_frame', {})
    if frame.get('observation_id') != obs['id']:
        return action('defer' if selected else 'explore', selected, 'No current grounded pair')
    ids = {}
    for detection in frame['detections']:
        for key in memory.observations[-1]['seen']:
            if any(e['observation_id']==obs['id'] and e['crop']==detection['crop'] for e in memory.instances[key]['evidence']):
                ids[detection['role']] = key
    target, reference = ids.get('target'), ids.get('reference')
    if not target or not reference or target == reference or target in excluded_ids(policy.task, memory.instances):
        policy.direct_relation_anchor = None
        return action('defer' if selected else 'explore', selected, frame['answer'].get('reason', 'No valid pair geometry'))
    from .near_navigation import relation_evidence
    if not relation_evidence(memory, target, reference, obs, final=at_goal):
        policy.direct_relation_anchor = None
        return action('defer' if selected else 'explore', selected, 'VLM pair lacks valid depth-based near geometry')
    policy.task.setdefault('near_references', {})[target] = reference
    reason = frame['answer']['reason']
    if not getattr(policy, 'direct_relation_anchor', None):
        policy.direct_relation_anchor = dict(observation_id=obs['id'],rgb_path=obs['rgb_path'],
            target=frame['answer']['target'],reference=frame['answer']['reference'])
    if at_goal and selected == target:
        return action('confirm', target, reason, reference_id=reference, reference_verified=True, reference_reason=reason)
    return action('select', target, reason, switch_reason=reason,
        visual_hint=dict(candidate_id=target, category_match='yes', attributes_match='yes',
                         reason=reason, observation_id=obs['id']))
