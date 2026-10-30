"""Measured cloud/Ollama JSON calls. Replay files never contain credentials."""
import json
import time
from pathlib import Path
from urllib import request, error
from urllib.parse import urlparse


def ollama_content(content):
    if isinstance(content, str):
        return dict(role='user', content=content)
    texts, images = [], []
    for part in content:
        if part['type'] == 'text':
            texts.append(part['text'])
        elif part['type'] == 'image_url':
            url = part['image_url']['url']
            if not url.startswith('data:image/') or ';base64,' not in url:
                raise ValueError('Ollama adapter requires embedded image data')
            images.append(url.split(',', 1)[1])
            texts.append('[Image %d follows the preceding label]' % len(images))
        else:
            raise ValueError('Unsupported multimodal content')
    return dict(role='user', content='\n'.join(texts), images=images)


def request_json(policy, system, content):
    policy.calls += 1
    images = sum(p.get('type') == 'image_url' for p in content) if isinstance(content, list) else 0
    metric = dict(call=policy.calls, backend=policy.backend, model=policy.model,
                  image_count=images, phase='decision' if images else 'instruction', ok=False)
    trace = Path(policy.trace_dir) if policy.trace_dir else None
    if trace:
        trace.mkdir(parents=True, exist_ok=True)
        (trace/('request_%04d.json' % policy.calls)).write_text(json.dumps(
            dict(system=system, content=content), ensure_ascii=False))
    headers = {'Content-Type': 'application/json'}
    if policy.key:
        headers['Authorization'] = 'Bearer '+policy.key
    messages = [dict(role='system', content=system), dict(role='user', content=content)]
    if policy.backend == 'ollama':
        messages[1] = ollama_content(content)
        payload = dict(model=policy.model, messages=messages, stream=False, think=False,
                       format='json', keep_alive=-1, options=dict(temperature=0,
                       num_ctx=policy.num_ctx, num_predict=policy.max_tokens))
        url = policy.base.removesuffix('/v1')+'/api/chat'
    else:
        payload = dict(model=policy.model, messages=messages, temperature=0, stream=False,
                       max_tokens=policy.max_tokens)
        if (urlparse(policy.base).hostname or '').endswith('aliyuncs.com') and policy.model.startswith('qwen'):
            payload.update(enable_thinking=False, response_format={'type':'json_object'})
        url = policy.base+'/chat/completions'
    begin = time.perf_counter()
    try:
        req = request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        # Never route local images through the user's HTTP proxy.
        open_url = request.build_opener(request.ProxyHandler({})).open if urlparse(url).hostname in {'localhost','127.0.0.1','::1'} else request.urlopen
        with open_url(req, timeout=policy.timeout) as response:
            packet = json.load(response)
        if policy.backend == 'ollama':
            for key in ('total_duration','load_duration','prompt_eval_duration','eval_duration'):
                if key in packet:
                    metric[key+'_seconds'] = packet[key]/1e9
            for key in ('prompt_eval_count','prompt_eval_cached_count','eval_count'):
                metric[key] = packet.get(key)
            raw = packet['message']['content']
            finish = packet.get('done_reason')
        else:
            choice = packet['choices'][0]
            raw, finish = choice['message']['content'], choice.get('finish_reason')
            metric['usage'] = packet.get('usage', {})
        metric['finish_reason'] = finish
        if trace:
            (trace/('response_%04d.json' % policy.calls)).write_text(json.dumps(packet, ensure_ascii=False))
        if finish != 'stop':
            raise ValueError('Incomplete VLM answer; no action executed')
        raw = raw.strip()
        if raw.startswith('```'):
            raw = raw.split('\n',1)[1].rsplit('```',1)[0]
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError('VLM response must be a JSON object')
        metric['ok'] = True
        return result
    except error.HTTPError as exc:
        metric['error'] = 'HTTP '+str(exc.code)
        raise RuntimeError('VLM HTTP %s; check model image support and configuration' % exc.code) from None
    except Exception as exc:
        metric['error'] = type(exc).__name__
        raise
    finally:
        metric['wall_seconds'] = time.perf_counter()-begin
        policy.metrics.append(metric)
        if trace:
            with (trace/'timings.jsonl').open('a') as file:
                file.write(json.dumps(metric, ensure_ascii=False)+'\n')
