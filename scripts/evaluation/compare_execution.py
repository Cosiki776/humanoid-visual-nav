#!/usr/bin/env python3
"""Run matched synchronous/asynchronous episodes sequentially on the same GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instruction',default='走到白色椅子旁边')
    p.add_argument('--scene',default='apt_0')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--vlm-model',default=os.environ.get('NAV_VLM_MODEL','qwen3.8-max'))
    p.add_argument('--max-observations',type=int,default=16)
    p.add_argument('--image-policy',choices=['adaptive','legacy'],default='adaptive')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gt',type=Path)
    p.add_argument('--episodes',type=Path,help='Independent evaluation file; choose matching instruction/scene')
    args=p.parse_args()
    if bool(args.gt)!=bool(args.episodes): p.error('--gt and --episodes must be supplied together')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]; start=None
    for mode in ('sync','async'):
        output=args.output/mode
        command=[sys.executable,str(Path(__file__).resolve().parents[2]/'run_visual.py'),
                 '--execution-mode',mode,'--image-policy',args.image_policy,'--realtime',
                 '--instruction',args.instruction,'--scene',args.scene,'--seed',str(args.seed),
                 '--vlm-model',args.vlm_model,'--max-observations',str(args.max_observations),
                 '--output',str(output)]
        if start is not None: command+=['--start']+[str(x) for x in start]
        begun=time.monotonic()
        with (args.output/(mode+'.log')).open('w') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        path=output/'result.json'
        if not path.exists():
            raise RuntimeError(mode+' produced no result; inspect '+str(args.output/(mode+'.log')))
        result=json.loads(path.read_text()); start=result['start_position'] if start is None else start
        row=dict(execution_mode=mode,process_seconds=time.monotonic()-begun,returncode=process.returncode,
                 status=result['status'],verified=result['verified'],path_length_m=result['path_length_m'],
                 elapsed_seconds=result['elapsed_seconds'],observations=result['observations'],
                 vlm_calls=result['vlm_calls'],start_position=result['start_position'],
                 background_observations=sum(bool(e.get('asynchronous')) for e in result['events']),
                 images_per_decision=[e['visual_input']['image_count'] for e in result['events']
                                      if e.get('visual_input')])
        if args.gt:
            from scripts.evaluation.evaluate_visual import score_episode
            gt=json.loads(args.gt.read_text()); episodes=json.loads(args.episodes.read_text())
            if isinstance(episodes,dict): episodes=[episodes]
            episode=next((e for e in episodes if e['instruction']==args.instruction and e['scene']==args.scene),None)
            if episode is None or gt['scene']!=args.scene: raise ValueError('Evaluation scene/instruction mismatch')
            row['evaluation']=score_episode(result,episode,gt['objects'])
        rows.append(row)
        (args.output/'report.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        print(json.dumps(row,ensure_ascii=False),flush=True)
    lines=['# 同起点同步/异步对照','',
           '两次任务顺序执行，包含实时行走。单次结果不代表成功率或稳定的加速比。','',
           '| 模式 | 状态 | 导航秒数 | 总进程秒数 | 路径米数 | 观察数 | 途中观察数 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {execution_mode} | {status} | {elapsed_seconds:.2f} | {process_seconds:.2f} | {path_length_m:.2f} | {observations} | {background_observations} |'.format(**r))
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
