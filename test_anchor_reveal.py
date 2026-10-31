"""Occluded target search moves without first needing a target detection."""
import tempfile
import unittest
import numpy as np
from visual_nav.inspection import rank_reveal_viewpoints
from visual_nav.memory import ObjectMemory
from visual_nav.loop import VisualNavigator
from test_visual_nav import detection, observation
from test_visual_ledger import task


class RevealTests(unittest.TestCase):
    def test_side_view_is_preferred_and_anchor_visibility_not_required(self):
        anchor=dict(bbox_min=[-.5,0,-.5],bbox_max=[.5,2,.5])
        current=np.array([0.,0,-2.])
        points=[np.array([0.,0,-3.]),np.array([2.,0,0.]),np.array([-2.,0,0.])]
        def visible(eye, point):
            # Segment versus opaque refrigerator AABB (slab intersection).
            delta=point-eye; lower,upper=0.,1.
            for i,(lo,hi) in enumerate(zip(anchor['bbox_min'],anchor['bbox_max'])):
                if abs(delta[i])<1e-9:
                    if eye[i]<lo or eye[i]>hi:return True
                else:
                    a,b=sorted(((lo-eye[i])/delta[i],(hi-eye[i])/delta[i]))
                    lower,upper=max(lower,a),min(upper,b)
            return lower>upper or upper<0 or lower>1
        ranked=rank_reveal_viewpoints(points,current,anchor,lambda p:[current,p],visible)
        self.assertTrue(ranked)
        self.assertNotEqual(float(ranked[0][0]),0.)
        self.assertTrue(all(abs(p[0])==2 for p in ranked))
        self.assertEqual(rank_reveal_viewpoints(points,current,anchor,lambda p:None,visible),[])
        self.assertEqual(rank_reveal_viewpoints(points,current,anchor,lambda p:[current,p],visible,points[1:]),[])

    def test_no_visible_space_or_long_detour_is_not_selected(self):
        anchor=dict(bbox_min=[-.5,0,-.5],bbox_max=[.5,2,.5]);current=np.array([0.,0,-2.]);points=[[2,0,0]]
        self.assertEqual(rank_reveal_viewpoints(points,current,anchor,lambda p:[current,p],lambda a,b:False),[])
        self.assertEqual(rank_reveal_viewpoints(points,current,anchor,lambda p:[current,[20,0,0],p],lambda a,b:True),[])

    def test_hidden_chair_is_discovered_only_after_lateral_movement(self):
        class IO:
            start=[0,0,0]
            def __init__(self):self.position=np.zeros(3);self.counter=0;self.moves=[]
            def search_yaw(self):return 0.  # Even an unfinished scan must not trap the robot here.
            def search_summary(self):return dict(completed_scans=0,complete=False)
            def reveal_options(self,anchor,visited):return [np.array([1.5,0,0])]
            def observe(self,**kwargs):
                self.counter+=1;obs=observation('o'+str(self.counter));obs['position']=self.position.tolist();return obs
            def docking_options(self,item):return [np.array([.8,0,-1.35])]
            def advance(self,goal,**kwargs):self.position=np.array(goal);self.moves.append(self.position.tolist());return dict(status='succeeded',reached=True)
            def trajectory(self):return [self.start]+self.moves
        class Detector:
            def detect(self,obs,*args,**kwargs):
                d=detection();d['category']='refrigerator'
                return [d]+([detection(.8,color='white')] if obs['position'][0]>.5 else [])
        class Policy:
            calls=0
            def decide(self,instruction,obs,memory,selected,at_goal):
                self.calls+=1
                if 'v0002' not in memory.instances:return dict(action='explore')
                return dict(action='confirm' if at_goal else 'select',candidate_id='v0002',evidence_id=obs['id'],reference_id='v0001',reference_verified=True,reference_reason='current refrigerator')
        m=ObjectMemory();d=detection();d['category']='refrigerator';m.integrate([d],observation('seed'))
        with tempfile.TemporaryDirectory() as tmp:
            io=IO();nav=VisualNavigator(io,Detector(),Policy(),tmp,task=task(),memory=m,max_observations=8)
            m.ledger.anchors['v0001']=dict(verified=True,category='refrigerator');m.ledger.active_anchor_id='v0001'
            result=nav.run('找冰箱旁边的白色椅子')
            self.assertTrue(result['verified'])
            self.assertEqual(io.moves[0],[1.5,0,0])
            self.assertEqual(result['observations'],3)
            self.assertTrue(any(e['type']=='anchor_local_search' for e in result['events']))


if __name__=='__main__':unittest.main()
