"""Task-scoped decisions over persistent visual evidence; no simulator semantics."""
import copy


def predicate(task):
    task = task or {}
    return {k: sorted(task.get(k, [])) for k in ('target_categories', 'attributes', 'reference_categories')} | {'relation': task.get('relation', '')}


class TaskLedger:
    def __init__(self):
        self.task_id = 0
        self.task = {}
        self.claims = []
        self.deferred = set()
        self.anchors = {}
        self.questions = []
        self.active_anchor_id = None

    def begin(self, task):
        self.task_id += 1
        self.task = copy.deepcopy(task or {})
        self.questions.clear()
        self.active_anchor_id = None
        self.deferred = set()

    def record(self, target, observation_id, reason, kind, anchor=None):
        # Missing evidence is uncertainty, never a permanent negative claim.
        if kind not in {'category_mismatch', 'attribute_mismatch', 'relation_mismatch'}:
            self.deferred.add(target)
            kind = 'insufficient_evidence'
        if (kind == 'relation_mismatch' and not anchor) or (kind == 'attribute_mismatch' and not self.task.get('attributes')):
            self.deferred.add(target)
            kind = 'insufficient_evidence'
        claim = dict(target_id=target, anchor_id=anchor, kind=kind, reason=reason,
                     observation_id=observation_id, predicate=predicate(self.task), task_id=self.task_id,
                     status='deferred' if kind == 'insufficient_evidence' else 'rejected',
                     revisit_condition='explicit_review_required; a new frame alone does not reopen')
        self.claims.append(claim)
        return claim

    def blocked(self, target, task=None, anchor=None):
        p = predicate(self.task if task is None else task)
        if target in self.deferred:
            return True
        for claim in self.claims:
            if claim['target_id'] != target or claim['status'] != 'rejected':
                continue
            old = claim['predicate']
            if old['target_categories'] != p['target_categories']:
                continue
            if claim['kind'] == 'category_mismatch':
                return True
            if claim['kind'] == 'attribute_mismatch' and old['attributes'] and old['attributes'] == p['attributes']:
                return True
            if claim['kind'] == 'relation_mismatch' and old['relation'] == p['relation'] and anchor == claim['anchor_id']:
                return True
        return False

    def anchor_verdict(self, key, category):
        assessment = self.anchors.get(key, {})
        if assessment.get('category', category) != category:
            return None
        return assessment.get('verified')

    def pairs(self, memory, task=None):
        task = self.task if task is None else task
        result = []
        for edge in memory.relations.summary():
            if edge['relation'] != task.get('relation') or edge['status'] == 'contradicted':
                continue
            directions = [(edge['source'], edge['target'])]
            if edge['relation'] == 'near':
                directions.append((edge['target'], edge['source']))
            for target, anchor in directions:
                a, b = memory.instances.get(target, {}), memory.instances.get(anchor, {})
                if a.get('category') not in task.get('target_categories', []) or b.get('category') not in task.get('reference_categories', []):
                    continue
                if (self.blocked(target, task, anchor) or anchor in self.deferred or
                        self.anchor_verdict(anchor, b.get('category')) is False):
                    continue
                result.append(dict(target_id=target, anchor_id=anchor, relation=edge['relation'],
                                   status='candidate', confidence=edge['confidence'],
                                   geometry=edge.get('geometry', {}), evidence_ids=[e['observation_id'] for e in edge['evidence'][-3:]],
                                   anchor_verified=self.anchor_verdict(anchor, b.get('category')) is True))
        seen = set(memory.observations[-1]['seen']) if memory.observations else set()
        return sorted(result, key=lambda p: (p['target_id'] not in seen or p['anchor_id'] not in seen, not p['anchor_verified'], -p['confidence']))

    def active_anchor(self, memory, task=None):
        task = self.task if task is None else task
        key = memory.resolve(self.active_anchor_id) if self.active_anchor_id else None
        item = memory.instances.get(key, {})
        if (item.get('category') in task.get('reference_categories', []) and
                key not in self.deferred and self.anchor_verdict(key, item['category']) is True):
            return key
        return None

    def stage_task(self, memory, task):
        if not task.get('relation') or self.active_anchor(memory, task):
            return copy.deepcopy(task)
        return dict(target_categories=list(task.get('reference_categories', [])), attributes=[],
                    reference_categories=[], relation='', another=False,
                    detection_categories=list(task.get('detection_categories', [])))

    def context(self, memory, task=None):
        task = self.task if task is None else task
        pairs = self.pairs(memory, task)
        anchors = [i['id'] for i in memory.instances.values()
                   if i['category'] in task.get('reference_categories', []) and i['id'] not in self.deferred and self.anchor_verdict(i['id'], i['category']) is not False]
        active = self.active_anchor(memory, task)
        if active:
            pairs = [pair for pair in pairs if pair['anchor_id'] == active]
        phase = ('SEARCH_TARGET_NEAR_ANCHOR' if active else 'SEARCH_ANCHOR') if task.get('relation') else 'SEARCH_TARGET'
        return dict(task_id=self.task_id, phase=phase, active_anchor_id=active, candidate_pairs=pairs[:3], anchor_ids=anchors[:3],
                    negative_evidence=[copy.deepcopy(c) for c in self.claims
                                       if self.blocked(c['target_id'], task, c['anchor_id']) and
                                       (c['status'] == 'rejected' or c['task_id'] == self.task_id)][-8:],
                    anchor_assessments={key: copy.deepcopy(value) for key, value in self.anchors.items()
                                        if key in memory.instances and memory.instances[key]['category'] in task.get('reference_categories', [])},
                    deferred_ids=sorted(self.deferred))

    def remap(self, alias, canonical):
        if self.active_anchor_id == alias:
            self.active_anchor_id = canonical
        for claim in self.claims:
            for field in ('target_id', 'anchor_id'):
                if claim[field] == alias:
                    claim[field] = canonical
        if alias in self.deferred:
            self.deferred.remove(alias)
            self.deferred.add(canonical)
        if alias in self.anchors:
            self.anchors.setdefault(canonical, self.anchors.pop(alias))

    def summary(self):
        return dict(task_id=self.task_id, active_anchor_id=self.active_anchor_id, task=copy.deepcopy(self.task), claims=copy.deepcopy(self.claims),
                    deferred_ids=sorted(self.deferred), anchors=copy.deepcopy(self.anchors), unresolved=list(self.questions))
