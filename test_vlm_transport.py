import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from visual_nav.transport import request_json, ollama_content


class TransportTests(unittest.TestCase):
    def policy(self, trace):
        return SimpleNamespace(calls=0, backend='ollama', model='local', key='',
            base='http://127.0.0.1:11434/v1', num_ctx=16384, max_tokens=1024,
            timeout=30, metrics=[], trace_dir=trace)

    def test_multimodal_translation_and_timing(self):
        content = [dict(type='text',text='current o1'), dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,YQ==')),
                   dict(type='text',text='v0001 from o1'),dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,Yg=='))]
        packet = dict(message=dict(content='{"action":"explore"}'),done_reason='stop',
                      total_duration=2_000_000_000,prompt_eval_count=500,eval_count=10)
        with tempfile.TemporaryDirectory() as tmp, patch('visual_nav.transport.request.build_opener') as opener:
            opener.return_value.open.return_value = io.BytesIO(json.dumps(packet).encode())
            p = self.policy(tmp)
            self.assertEqual(request_json(p,'rules',content)['action'],'explore')
            req = opener.return_value.open.call_args.args[0]
            body = json.loads(req.data)
            self.assertEqual(req.full_url,'http://127.0.0.1:11434/api/chat')
            self.assertFalse(body['think'])
            self.assertEqual(body['keep_alive'], -1)
            self.assertEqual(body['options']['num_ctx'],16384)
            self.assertEqual(body['messages'][1]['images'],['YQ==','Yg=='])
            self.assertEqual(p.metrics[-1]['total_duration_seconds'],2)
            self.assertEqual(p.metrics[-1]['image_count'],2)
            saved = json.loads((Path(tmp)/'request_0001.json').read_text())
            self.assertEqual(saved,dict(system='rules',content=content))

    def test_truncation_and_bad_json_record_failure(self):
        for finish, content in [('length','{}'),('stop','not json')]:
            with self.subTest(finish=finish), patch('visual_nav.transport.request.build_opener') as opener:
                opener.return_value.open.return_value=io.BytesIO(json.dumps(dict(message=dict(content=content),done_reason=finish)).encode())
                p=self.policy(None)
                with self.assertRaises(ValueError): request_json(p,'rules','hello')
                self.assertFalse(p.metrics[-1]['ok'])
                self.assertIn('wall_seconds',p.metrics[-1])

    def test_no_remote_image_fetch(self):
        with self.assertRaises(ValueError):
            ollama_content([dict(type='image_url',image_url=dict(url='https://example.com/image.jpg'))])

    def test_request_lifecycle_identifies_actual_frame(self):
        events=[]
        p=self.policy(None)
        p.request_callback=events.append
        content=[dict(type='text',text=json.dumps(dict(observation_id='o7',candidate_id='v0018'))),
                 dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,YQ=='))]
        packet=dict(message=dict(content='{"approach_recommended":true}'),done_reason='stop')
        with patch('visual_nav.transport.request.build_opener') as opener:
            def response(*args, **kwargs):
                self.assertEqual(events[-1]['status'],'waiting')
                return io.BytesIO(json.dumps(packet).encode())
            opener.return_value.open.side_effect=response
            request_json(p,'rules',content)
        self.assertEqual([e['status'] for e in events],['waiting','returned'])
        self.assertEqual(events[0]['observation_id'],'o7')
        self.assertEqual(events[0]['image_count'],1)
        self.assertTrue(events[1]['answer']['approach_recommended'])

    def test_request_error_is_reported(self):
        events=[]
        p=self.policy(None)
        p.request_callback=events.append
        with patch('visual_nav.transport.request.build_opener') as opener:
            opener.return_value.open.side_effect=TimeoutError()
            with self.assertRaises(TimeoutError): request_json(p,'rules','hello')
        self.assertEqual([e['status'] for e in events],['waiting','error'])

    def test_preload_matches_context_and_reports_errors(self):
        from warmup_vlm import preload
        with patch('warmup_vlm.request.build_opener') as opener:
            opener.return_value.open.return_value = io.BytesIO(b'{"done":true,"load_duration":2000000000}')
            result = preload('http://127.0.0.1:11434/v1', 'local', 16384)
            self.assertEqual(result['load_seconds'], 2)
            body = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertEqual(body['messages'], [])
            self.assertEqual(body['keep_alive'], -1)
            self.assertEqual(body['options']['num_ctx'], 16384)
            opener.return_value.open.return_value = io.BytesIO(b'{"error":"missing model"}')
            with self.assertRaises(RuntimeError):
                preload('http://127.0.0.1:11434', 'local')
