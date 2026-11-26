import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from PIL import Image
from tests.test_visual_nav import observation, detection
from visual_nav.direct_relation import locate_pair, DirectRelationPerception, decide_direct
from visual_nav.memory import ObjectMemory
from visual_nav.policy import VLMPolicy, DecisionValidationError
from visual_nav.loop import VisualNavigator
import numpy as np


class DirectRelationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'frame.jpg'
        Image.new('RGB',(768,512),'white').save(self.path)
        self.obs=observation('o1')
        self.obs['rgb_path']=str(self.path)
        self.task=dict(target_categories=['chair'],reference_categories=['refrigerator'],attributes=['white'],relation='near',relation_source='vlm_grounding')
        self.answer=dict(observation_id='o1',found=True,reason='White chair beside fridge',
            target=dict(category='chair',bbox=[100,200,300,600]),reference=dict(category='refrigerator',bbox=[400,100,700,900]))
        self.policy=VLMPolicy.__new__(VLMPolicy)
        self.policy.calls=0
        self.policy.task=self.task
        self.policy._request=Mock(side_effect=lambda *args:dict(self.answer))

    def test_raw_image_only_and_pixel_conversion(self):
        answer,boxes=locate_pair(self.policy,self.obs,self.task)
        context=json.loads(self.policy._request.call_args.args[1][0]['text'])
        self.assertNotIn('visible_candidates',context)
        self.assertEqual(len(self.policy._request.call_args.args[1]),2)
        self.assertAlmostEqual(boxes[0]['bbox'][0],76.8)
        self.assertAlmostEqual(boxes[0]['bbox'][3],307.2)

    def test_no_match_does_not_segment(self):
        self.answer['found']=False
        segmenter=Mock()
        wrapper=DirectRelationPerception(segmenter,self.policy,self.task)
        self.assertEqual(wrapper.detect(self.obs,self.tmp.name),[])
        segmenter.segment_boxes.assert_not_called()
        segmenter.detect.assert_not_called()

    def test_invalid_box_cannot_become_navigation(self):
        self.answer['target']['bbox']=[0,0,float('nan'),1000]
        with self.assertRaises(DecisionValidationError):locate_pair(self.policy,self.obs,self.task)

    def test_depth_pair_selects_and_current_arrival_confirms_without_second_call(self):
        memory=ObjectMemory()
        detections=[]
        for role,offset,category in [('target',0,'chair'),('reference',.8,'refrigerator')]:
            d=detection(offset)
            d.update(role=role,category=category,crop=str(self.path)+role,bbox=[10,10,30,30])
            detections.append(d)
        memory.integrate(detections,self.obs)
        self.policy.direct_relation_frame=dict(observation_id='o1',answer=self.answer,detections=detections)
        self.assertEqual(decide_direct(self.policy,self.obs,memory,None,False)['action'],'select')
        result=decide_direct(self.policy,self.obs,memory,'v0001',True)
        self.assertEqual(result['action'],'confirm')
        self.policy._request.assert_not_called()

    def test_missing_depth_reference_cannot_select(self):
        memory=ObjectMemory()
        d=detection();d.update(role='target',crop=str(self.path))
        memory.integrate([d],self.obs)
        self.policy.direct_relation_frame=dict(observation_id='o1',answer=self.answer,detections=[d])
        self.assertEqual(decide_direct(self.policy,self.obs,memory,None,False)['action'],'explore')

    def test_controller_uses_grounding_once_then_fresh_arrival_frame(self):
        task=self.task
        task['detection_categories']=['chair','refrigerator']
        self.policy.program_control=True
        self.policy._request.side_effect=lambda s,c: dict(self.answer,
            observation_id=json.loads(c[0]['text'])['observation_id'])
        outer=self
        class IO:
            start=[0,0,0]
            def __init__(self): self.position=np.zeros(3); self.n=0
            def observe(self,**kwargs):
                self.n+=1
                obs=observation('o'+str(self.n))
                obs.update(rgb_path=str(outer.path),position=self.position.tolist())
                return obs
            def docking_options(self,item):return [np.array([0.,0.,-1.35])]
            def advance(self,goal,**kwargs):
                self.position=np.array(goal)
                return dict(status='succeeded',reached=True,travelled=1.35)
            def trajectory(self):return [[0,0,0],self.position.tolist()]
        class Segmenter:
            def detect(self,*args,**kwargs):raise AssertionError('DINO must not run')
            def segment_boxes(self,obs,output,boxes):
                detections=[]
                for index,b in enumerate(boxes):
                    d=detection(index*.8)
                    d.update(role=b['role'],category=b['category'],bbox=b['bbox'],
                             crop=str(outer.path)+b['role'])
                    detections.append(d)
                return detections
        nav=VisualNavigator(IO(),Segmenter(),self.policy,self.tmp.name,task=task,max_observations=5)
        result=nav.run('去冰箱旁边的白色椅子旁边')
        self.assertEqual(result['status'],'succeeded')
        self.assertEqual(self.policy._request.call_count,2)
