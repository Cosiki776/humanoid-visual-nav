"""Single-flight background inference; simulator and memory writes stay on owner."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor


def semantic_signature(memory, task, selected):
    categories = set((task or {}).get('target_categories', [])) | set((task or {}).get('reference_categories', []))
    items = []
    for key, item in memory.instances.items():
        if categories and item['category'] not in categories and key != selected:
            continue
        if key != selected and item.get('views', 0) < 2:
            continue
        scores = item.get('attributes', {}).get('color_scores', {})
        color = max(scores, key=scores.get) if scores else None
        items.append((key, item['category'], color))
    relevant = {i[0] for i in items}
    edges = [(e['source'], e['relation'], e['target'], e['status']) for e in memory.relations.summary()
             if (task or {}).get('relation') == e['relation'] and e['source'] in relevant and e['target'] in relevant]
    return (tuple(sorted(items)), tuple(sorted(edges)))


def worthwhile_sample(remaining_metres, perception_seconds, walking_speed=.8):
    """Avoid launching work unlikely to finish before the mandatory end observation."""
    return remaining_metres/walking_speed >= perception_seconds+.3


class AsyncInference:
    """One detection and one policy job maximum; no unbounded frame queue."""
    def __init__(self, detector, policy, output, categories):
        self.detector, self.policy = detector, policy
        self.output, self.categories = output, categories
        self.perception_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='visual-perception')
        self.reasoning_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='visual-reasoning')
        self.perception = self.reasoning = None
        self.closed = False

    def submit_observation(self, observation):
        if self.closed or self.perception is not None:
            return False
        # Copy RGB-D/pose arrays before the owner renders the next simulator frame.
        obs = copy.deepcopy(observation)
        def work():
            started = time.perf_counter()
            detections = self.detector.detect(obs, self.output, categories=self.categories)
            return obs, detections, time.perf_counter()-started
        self.perception = self.perception_pool.submit(work)
        return True

    def take_observation(self, wait=False):
        if self.perception is None or (not wait and not self.perception.done()):
            return None
        job, self.perception = self.perception, None
        return job.result()

    def submit_decision(self, instruction, obs, memory, selected):
        if self.closed or self.reasoning is not None:
            return False
        snapshot = copy.deepcopy(memory)
        observation = copy.deepcopy(obs)
        self.reasoning = self.reasoning_pool.submit(self.policy.decide, instruction,
                                                    observation, snapshot, selected, False)
        return True

    def take_decision(self, wait=False):
        if self.reasoning is None or (not wait and not self.reasoning.done()):
            return None
        job, self.reasoning = self.reasoning, None
        return job.result()

    def close(self):
        self.closed = True
        # Drain before a session can reuse the detector/policy or switch task.
        self.perception_pool.shutdown(wait=True, cancel_futures=True)
        self.reasoning_pool.shutdown(wait=True, cancel_futures=True)
