"""Deterministic relative motion commands; no visual model or route planner."""
import math
import re
import time
import numpy as np


def parse_directional(instruction):
    match = re.fullmatch(r'(?:请)?(?:机器人)?\s*(向前|往前|前进|向后|往后|后退|向左|往左|左移|向右|往右|右移)\s*(?:移动|走|前进|后退)?\s*([+-]?\d+(?:\.\d+)?|[零一二两三四五六七八九十百点半]+)\s*(?:米|m)\s*[。！!]?\s*', instruction.strip(), re.I)
    if not match:
        return None
    direction, number = match.groups()
    digits = dict(zip('零一二两三四五六七八九', [0,1,2,2,3,4,5,6,7,8,9]))
    def integer(s):
        if '百' in s:
            raise ValueError('定向移动距离请设为0.01至20米')
        if '十' in s:
            left,right=s.split('十')
            return (digits[left] if left else 1)*10+(digits[right] if right else 0)
        return int(''.join(str(digits[c]) for c in s))
    try:
        if number == '半': distance=.5
        elif '点' in number:
            whole,fraction=number.split('点')
            distance=integer(whole)+float('0.'+''.join(str(digits[c]) for c in fraction))
        elif number[0] in digits or '十' in number or '百' in number: distance=float(integer(number))
        else: distance=float(number)
    except (ValueError, KeyError):
        raise ValueError('定向移动距离请使用0.01至20米，例如“向前走2米”') from None
    if not math.isfinite(distance) or not .01 <= distance <= 20:
        raise ValueError('定向移动距离请设为0.01至20米')
    key = 'forward' if '前' in direction else 'backward' if '后' in direction else 'left' if '左' in direction else 'right'
    return dict(kind='directional_move', direction=key, distance_m=distance)


def direction_vector(forward, direction):
    forward=np.asarray(forward,dtype=float).copy()
    forward[1]=0
    if not np.isfinite(forward).all() or np.linalg.norm(forward)<1e-6:
        raise ValueError('无法确定机器人当前朝向')
    forward/=np.linalg.norm(forward)
    right=np.array([-forward[2],0,forward[0]])
    return {'forward':forward,'backward':-forward,'left':-right,'right':right}[direction]


def move_straight(executor, world, direction, distance, should_stop=lambda:False, progress=None):
    start=executor.position.copy()
    vector=direction_vector(executor.controller.prev_orientation,direction)
    travelled=0.
    status='succeeded'
    reason='requested distance reached'
    # Find a safe straight prefix, not a snapped destination or a detour.
    safe=0.
    for step in range(1,int(math.ceil(distance/.05))+1):
        if should_stop():
            status,reason='cancelled','user cancelled';break
        amount=min(distance,step*.05)
        if not world.direct(start+vector*safe,start+vector*amount):
            break
        safe=amount
    if status!='cancelled' and safe>.005:
        event=executor.turn_to(vector,should_stop)
        status=event['status']
        reason=event.get('reason',reason)
        while status=='succeeded':
            along=float(np.dot(executor.position-start,vector))
            if safe-along<=.01:break
            endpoint=start+vector*min(safe,along+.25)
            before=executor.position.copy()
            event=executor.execute_segment(endpoint,should_stop=should_stop,arrival_tolerance=.01,max_steps=180)
            travelled+=float(np.linalg.norm(executor.position-before))
            status=event['status'];reason=event.get('reason',reason)
            if progress:progress(travelled)
            if status=='succeeded' and np.linalg.norm(executor.position-before)<.001:
                status,reason='blocked','no progress'
    if status=='succeeded' and safe<distance-1e-6:
        status,reason='blocked','obstacle or non-traversable NavMesh boundary'
    executor.stop()
    return dict(status=status,reason=reason,requested_distance_m=distance,travelled=travelled,
                displacement_m=float(np.linalg.norm(executor.position-start)),
                start_position=start.tolist(),position=executor.position.tolist(),
                direction_vector=vector.tolist())


def run_directional(simulator, task, instruction, memory, emit):
    begin=time.monotonic()
    def progress(travelled):
        emit(dict(type='live_state',phase='moving',task=task,observations=0,
                  path_length_m=travelled,position=simulator.position.tolist()))
    event=move_straight(simulator._executor,simulator._world,task['direction'],task['distance_m'],simulator.should_stop,progress)
    vector=np.asarray(simulator._executor.controller.prev_orientation)
    simulator.yaw=math.atan2(vector[0],-vector[2])
    simulator.visited.append(simulator.position.tolist())
    distance=event['travelled']
    message=(f'已移动{distance:.2f}米。' if event['status']=='succeeded' else
             f'遇到障碍物或不可通行边界，已停止；实际移动{distance:.2f}米，未绕路。' if event['status']=='blocked' else
             f'移动已停止（{event["status"]}），实际移动{distance:.2f}米。')
    return dict(status=event['status'],mode='directional',policy='DeterministicMotion',task=task,
        instruction=instruction,message=message,observations=0,vlm_calls=0,verified=False,selected=None,
        requested_distance_m=task['distance_m'],path_length_m=distance,
        start_position=event['start_position'],final_position=event['position'],elapsed_seconds=time.monotonic()-begin,
        events=[dict(type='execution',**event)],memory=memory.summary())
