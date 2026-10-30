"""Regression cases for the repeated-chair loop and scoped negative evidence."""
import tempfile
import unittest
from visual_nav.memory import ObjectMemory
from visual_nav.loop import VisualNavigator
from visual_nav.ledger import TaskLedger
from test_visual_nav import detection, observation


def task(color='white', relation='near'):
    return dict(target_categories=['chair'], attributes=[color] if color else [],
                reference_categories=['refrigerator'] if relation else [], relation=relation,
                detection_categories=['chair', 'refrigerator'])


class LedgerTests(unittest.TestCase):
    def test_attribute_rejection_survives_next_matching_task_but_not_plain_chair(self):
        ledger = TaskLedger()
        ledger.begin(task())
        ledger.record('a', 'o1', 'brown', 'attribute_mismatch')
        ledger.begin(task())
        self.assertTrue(ledger.blocked('a'))
        self.assertFalse(ledger.blocked('b'))
        ledger.begin(task(None, ''))
        self.assertFalse(ledger.blocked('a'))

    def test_pair_rejection_does_not_reject_other_anchor(self):
        ledger = TaskLedger()
        ledger.begin(task())
        ledger.record('a', 'o1', 'far from x', 'relation_mismatch', 'x')
        self.assertTrue(ledger.blocked('a', anchor='x'))
        self.assertFalse(ledger.blocked('a', anchor='y'))
        self.assertFalse(ledger.blocked('a'))

    def test_absence_is_deferred_not_permanent_negative(self):
        ledger = TaskLedger()
        ledger.begin(task())
        claim = ledger.record('a', 'o1', 'anchor not visible', 'insufficient_evidence')
        self.assertEqual(claim['status'], 'deferred')
        self.assertTrue(ledger.blocked('a'))
        ledger.begin(task())
        self.assertFalse(ledger.blocked('a'))

    def test_category_revision_keeps_identity_and_negative_evidence(self):
        m = ObjectMemory()
        m.ledger.begin(task())
        key = m.integrate([detection()], observation())[0]
        m.reject(key, 'o1', 'brown', 'attribute_mismatch')
        revised = detection()
        revised['category'] = 'stool'
        self.assertEqual(m.integrate([revised], observation('o2')), [key])
        self.assertEqual(len(m.instances), 1)
        self.assertTrue(m.ledger.blocked(key))
        self.assertEqual(m.instances[key]['category'], 'stool')

    def test_reappearance_does_not_clear_rejection_or_merge_neighbor(self):
        m = ObjectMemory()
        m.ledger.begin(task())
        a, b = m.integrate([detection(), detection(.6)], observation())
        m.reject(a, 'o1', 'brown', 'attribute_mismatch')
        ids = m.integrate([detection(.03), detection(.61)], observation('o2'))
        self.assertEqual(ids, [a, b])
        self.assertTrue(m.ledger.blocked(a))
        self.assertFalse(m.ledger.blocked(b))

    def test_guard_blocks_unrelated_chair_and_keeps_anchor_inspection(self):
        m = ObjectMemory()
        a = m.integrate([detection()], observation())[0]
        with tempfile.TemporaryDirectory() as tmp:
            nav = VisualNavigator(object(), object(), object(), tmp, task=task(), memory=m)
            self.assertEqual(nav.guard_decision(dict(action='select', candidate_id=a), None, False)['action'], 'explore')
            fridge = detection(1.)
            fridge['category'] = 'refrigerator'
            m.integrate([detection(), fridge], observation('o2'))
            pairs = m.ledger.pairs(m)
            self.assertEqual(len(pairs), 1)
            anchor = pairs[0]['anchor_id']
            self.assertEqual(nav.guard_decision(dict(action='observe', candidate_id=anchor), None, False)['action'], 'observe')
            pending = nav.guard_decision(dict(action='select', candidate_id=a), None, False)
            self.assertEqual(pending['candidate_id'], anchor)
            m.ledger.anchors[anchor] = dict(verified=True)
            selected = nav.guard_decision(dict(action='select', candidate_id=a), None, False)
            self.assertEqual(selected['reference_id'], anchor)
            m.ledger.record(a, 'o2', 'wrong relation', 'relation_mismatch', anchor)
            self.assertEqual(nav.guard_decision(dict(action='select', candidate_id=a), None, False)['action'], 'explore')

    def test_alias_migration_preserves_rejection_and_never_reuses_id(self):
        m = ObjectMemory()
        m.ledger.begin(task())
        a = m.integrate([detection()], observation())[0]
        m.reject(a, 'o1', 'brown', 'attribute_mismatch')
        b = m.integrate([detection(2)], observation('o2'))[0]
        m.merge_alias(a, b)
        self.assertEqual(m.resolve(a), b)
        self.assertTrue(m.ledger.blocked(b))
        self.assertEqual(m.rejections[0]['id'], b)
        self.assertEqual(m.observations[0]['seen'], [b])
        c = m.integrate([detection(4)], observation('o3'))[0]
        self.assertNotIn(c, {a, b})

    def test_coobserved_neighbors_cannot_be_aliased(self):
        m = ObjectMemory()
        a, b = m.integrate([detection(), detection(.4)], observation())
        with self.assertRaises(ValueError):
            m.merge_alias(a, b)

    def test_anchor_assessment_requires_current_image_and_id(self):
        m = ObjectMemory()
        f = detection(1)
        f['category'] = 'refrigerator'
        a, anchor = m.integrate([detection(), f], observation())
        with tempfile.TemporaryDirectory() as tmp:
            nav = VisualNavigator(object(), object(), object(), tmp, task=task(), memory=m)
            d = dict(action='explore', anchor_assessment=dict(id=anchor, matches=False, evidence_id='old', reason='cabinet'))
            nav.guard_decision(d, None, False)
            self.assertNotIn(anchor, m.ledger.anchors)
            d['anchor_assessment']['evidence_id'] = 'o1'
            nav.guard_decision(d, None, False)
            self.assertEqual(m.ledger.pairs(m), [])
            m.ledger.begin(task())
            self.assertEqual(m.ledger.pairs(m), [])

    def test_anchor_negative_is_scoped_to_its_category(self):
        ledger = TaskLedger()
        ledger.anchors['a'] = dict(category='refrigerator', verified=False)
        self.assertFalse(ledger.anchor_verdict('a', 'refrigerator'))
        self.assertIsNone(ledger.anchor_verdict('a', 'cabinet'))

    def test_indirect_near_is_not_a_candidate_pair(self):
        m = ObjectMemory()
        m.ledger.begin(task())
        ds = [detection(0), detection(1), detection(2)]
        ds[1]['category'] = 'table'
        ds[2]['category'] = 'refrigerator'
        m.integrate(ds, observation())
        self.assertEqual(m.ledger.pairs(m), [])


if __name__ == '__main__':
    unittest.main()
