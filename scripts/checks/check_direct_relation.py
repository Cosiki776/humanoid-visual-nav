"""Test raw-image VLM grounding and optionally SAM/depth on recorded frames."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from visual_nav.policy import VLMPolicy
from visual_nav.direct_relation import locate_pair
from visual_nav.annotations import draw_labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',nargs='+',default=['obs_00020','obs_00021'])
    p.add_argument('--segment',action='store_true')
    p.add_argument('--sequence',action='store_true',help='Use first found frame as identity context for later frames')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    saved=json.loads((args.run/'result.json').read_text())
    policy=VLMPolicy()
    policy.trace_dir=args.output/'vlm_requests'
    task=saved['task']
    task['instruction']=saved['instruction']
    segmenter=None
    if args.segment:
        import bootstrap
        from segment_anything import sam_model_registry, SamPredictor
        from visual_nav.perception import GroundedSAM
        segmenter=GroundedSAM.__new__(GroundedSAM)
        segmenter.sam=SamPredictor(sam_model_registry['vit_b'](
            checkpoint=str(bootstrap.WORKSPACE/'weights/sam_vit_b_01ec64.pth')).to('cuda:0').eval())
    results=[]
    for key in args.frames:
        with np.load(args.run/(key+'.npz')) as data:
            obs=dict(data)
        obs.update(id=key,rgb_path=str(args.run/(key+'.jpg')))
        obs['rgb']=np.asarray(Image.open(obs['rgb_path']).convert('RGB'))
        answer,boxes=locate_pair(policy,obs,task)
        if args.sequence and answer['found'] and not getattr(policy,'direct_relation_anchor',None):
            policy.direct_relation_anchor=dict(observation_id=key,rgb_path=obs['rgb_path'],target=answer['target'],reference=answer['reference'])
        preview=Image.fromarray(obs['rgb'])
        draw_labels(preview,[dict(id=b['role'],category=b['category'],bbox=b['bbox']) for b in boxes],selected='target')
        preview.save(args.output/(key+'_boxes.jpg'))
        result=dict(observation_id=key,answer=answer,pixel_boxes=boxes)
        if segmenter:
            detections=segmenter.segment_boxes(obs,args.output,boxes)
            result['segmented']=[dict(role=d['role'],points=len(d['points']),position=np.median(d['points'],axis=0).tolist()) for d in detections]
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        (args.output/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
