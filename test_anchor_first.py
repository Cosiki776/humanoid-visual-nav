"""Anchor subtask is independently executable and never completes the whole task."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from PIL import Image
from visual_nav.memory import ObjectMemory
from visual_nav.loop import VisualNavigator
from visual_nav.policy import VLMPolicy, DecisionValidationError
from test_visual_nav import detection, observation
from test_visual_ledger import task


class AnchorFirstTests(unittest.TestCase):
    def test_vlm_first_stage_is_anchor_only_even_without_chair(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'image.jpg'; Image.new('RGB',(16,16)).save(image)
            m = ObjectMemory(); m.ledger.begin(task())
            d = detection(); d['category']='refrigerator'; d['crop']=str(image)
            obs = observation(); obs['rgb_path']=str(image)
            m.integrate([d], obs)
            policy = VLMPolicy.__new__(VLMPolicy); policy.task=task()
            policy._request=Mock(return_value=dict(action='select', candidate_id='v0001'))
            policy.decide('找冰箱旁边的白色椅子', obs, m)
            context=json.loads(policy._request.call_args.args[1][0]['text'])
            self.assertEqual(context['task']['target_categories'], ['refrigerator'])
            self.assertEqual(context['task']['attributes'], [])
            self.assertEqual(context['task']['relation'], '')
            self.assertEqual(context['task_memory']['candidate_pairs'], [])
            self.assertEqual(context['task_memory']['phase'], 'SEARCH_ANCHOR')
            self.assertEqual(policy.task, task())  # No shared policy mutation during async inference.

    def test_confirm_anchor_without_pair_transitions_but_does_not_confirm_task(self):
        m=ObjectMemory(); d=detection(); d['category']='refrigerator'
        anchor=m.integrate([d], observation())[0]
        with tempfile.TemporaryDirectory() as tmp:
            nav=VisualNavigator(object(), object(), object(), tmp, task=task(), memory=m)
            decision=dict(action='confirm', candidate_id=anchor, evidence_id='o1', reason='refrigerator door visible')
            with self.assertRaises(DecisionValidationError): nav.guard_decision(decision, anchor, False)
            result=nav.guard_decision(decision, anchor, True)
            self.assertEqual(result['action'], 'explore')
            self.assertEqual(m.ledger.active_anchor(m), anchor)
            self.assertEqual(m.ledger.context(m)['phase'], 'SEARCH_TARGET_NEAR_ANCHOR')
            self.assertEqual(m.ledger.pairs(m), [])
            self.assertEqual(m.ledger.stage_task(m,task())['target_categories'], ['chair'])

    def test_closed_loop_confirms_anchor_then_chair(self):
        class IO:
            start=[0,0,0]
            def __init__(self): self.position=np.zeros(3); self.counter=0
            def observe(self, **kwargs):
                self.counter+=1
                obs=observation('o'+str(self.counter)); obs['position']=self.position.tolist()
                return obs
            def docking_options(self,item): return [np.array([item['position'][0],0,-1.35])]
            def advance(self, goal, **kwargs): self.position=np.array(goal); return dict(status='succeeded',reached=True)
            def trajectory(self): return [self.start,self.position.tolist()]
        class Detector:
            def detect(self,obs,*args,**kwargs):
                fridge=detection(); fridge['category']='refrigerator'
                return [fridge] if obs['id']=='o1' else [fridge,detection(.8,color='white')]
        class Policy:
            calls=0
            def decide(self,instruction,obs,memory,selected,at_goal):
                self.calls+=1
                key='v0001' if not memory.ledger.active_anchor(memory) else 'v0002'
                d=dict(action='confirm' if selected==key and at_goal else 'select', candidate_id=key,
                       evidence_id=obs['id'],reason='visible category and attribute')
                if key=='v0002': d.update(reference_id='v0001',reference_verified=True,reference_reason='refrigerator visible')
                return d
        with tempfile.TemporaryDirectory() as tmp:
            result=VisualNavigator(IO(),Detector(),Policy(),tmp,task=task(),max_observations=8).run('找冰箱旁边的白色椅子')
            self.assertTrue(result['verified'])
            self.assertEqual(result['selected'],'v0002')
            self.assertEqual(sum(e['type']=='anchor_confirmed' for e in result['events']),1)
            self.assertEqual(result['observations'],4)

    def test_reject_anchor_records_negative_and_remains_in_anchor_stage(self):
        m=ObjectMemory();d=detection();d['category']='refrigerator';key=m.integrate([d],observation())[0]
        with tempfile.TemporaryDirectory() as tmp:
            nav=VisualNavigator(object(),object(),object(),tmp,task=task(),memory=m)
            decision=dict(action='reject',candidate_id=key,evidence_id='o1',reason='visible wooden cabinet')
            self.assertEqual(nav.guard_decision(decision,key,True)['action'],'explore')
            self.assertIsNone(m.ledger.active_anchor(m))
            self.assertFalse(m.ledger.anchor_verdict(key,'refrigerator'))
            self.assertEqual(nav.guard_decision(dict(action='select',candidate_id=key),None,False)['action'],'explore')

    def test_pinned_anchor_prevents_switch_to_other_pair(self):
        m=ObjectMemory()
        ds=[detection(0),detection(.8),detection(1.6)]
        ds[0]['category']=ds[2]['category']='refrigerator'
        a,target,b=m.integrate(ds,observation())
        with tempfile.TemporaryDirectory() as tmp:
            nav=VisualNavigator(object(),object(),object(),tmp,task=task(),memory=m)
            for key in (a,b): m.ledger.anchors[key]=dict(verified=True,category='refrigerator')
            m.ledger.active_anchor_id=a
            wrong=nav.guard_decision(dict(action='select',candidate_id=target,reference_id=b),None,False)
            self.assertEqual(wrong['action'],'explore')
            correct=nav.guard_decision(dict(action='select',candidate_id=target,reference_id=a),None,False)
            self.assertEqual(correct['action'],'select')

    def test_local_search_is_bounded_and_does_not_use_global_goal(self):
        class IO:
            position=np.zeros(3)
            def explore_goal(self): raise AssertionError('must search around pinned anchor')
            def inspection_options(self,*args): return [self.position+np.array([1.,0,0])]
        m=ObjectMemory();d=detection();d['category']='refrigerator';key=m.integrate([d],observation())[0]
        with tempfile.TemporaryDirectory() as tmp:
            io=IO();nav=VisualNavigator(io,object(),object(),tmp,task=task(),memory=m)
            m.ledger.anchors[key]=dict(verified=True,category='refrigerator');m.ledger.active_anchor_id=key
            for _ in range(3): io.position=nav.next_search_goal()
            self.assertIsNone(nav.next_search_goal())


if __name__=='__main__': unittest.main()
