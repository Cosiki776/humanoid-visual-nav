#!/usr/bin/env python3
"""Preload the existing Ollama service; never start or reassign its GPU."""
import json
import os
import time
from urllib import request


def preload(base, model, num_ctx=16384, timeout=300):
    payload = dict(model=model, messages=[], stream=False, think=False,
                   keep_alive=-1, options=dict(num_ctx=num_ctx))
    req = request.Request(base.rstrip('/').removesuffix('/v1')+'/api/chat',
                          data=json.dumps(payload).encode(),
                          headers={'Content-Type':'application/json'})
    opener = request.build_opener(request.ProxyHandler({}))
    started = time.perf_counter()
    with opener.open(req, timeout=timeout) as response:
        packet = json.load(response)
    if packet.get('error') or packet.get('done') is not True:
        raise RuntimeError('Ollama preload did not complete; navigation server not started')
    return dict(model=model, num_ctx=num_ctx, keep_alive=-1,
                elapsed_seconds=time.perf_counter()-started,
                load_seconds=packet.get('load_duration', 0)/1e9)


def main():
    model = os.environ.get('NAV_VLM_MODEL', 'qwen3.8:27b-q4_K_M')
    print('正在预加载本地模型 '+model+'，完成后再启动导航网页……', flush=True)
    result = preload(os.environ.get('NAV_VLM_BASE_URL', 'http://127.0.0.1:11434'),
                     model, int(os.environ.get('NAV_VLM_NUM_CTX', '16384')),
                     float(os.environ.get('NAV_VLM_TIMEOUT', '300')))
    print('本地模型已加载，保持驻留：'+json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
