import io
import json
import threading
import time
import unittest
from unittest.mock import patch

from planning import ModelPlanner
from serve import LiveApp


class ObservabilityTests(unittest.TestCase):
    def planner(self, events):
        planner = ModelPlanner.__new__(ModelPlanner)
        planner.base, planner.model, planner.key = 'https://example.com/v1', 'test', 'secret'
        planner.timeout, planner.emit = 10, events.append
        return planner

    def test_request_records_only_safe_metadata(self):
        events = []
        response = io.BytesIO(('data: '+json.dumps({'choices': [{'delta': {'content': '{"route": []}'},
                                                                 'finish_reason': 'stop'}]})+'\n\ndata: [DONE]\n\n').encode())
        with patch('planning.request.urlopen', return_value=response):
            self.assertEqual(self.planner(events)._request('system', {'scene': {}}), {'route': []})
        self.assertEqual([e['state'] for e in events if e['type'] == 'model_request'], ['started', 'succeeded'])
        self.assertEqual(events[0]['provider'], 'example.com')
        self.assertEqual(events[1]['type'], 'provider_response')
        self.assertNotIn('private', json.dumps(events))
        self.assertNotIn('secret', json.dumps(events))

    def test_stream_reasoning_is_display_only(self):
        events = []
        chunks = [{'choices': [{'delta': {'reasoning_content': '公开的推理文本'}}]},
                  {'choices': [{'delta': {'content': '{"route":'}}]},
                  {'choices': [{'delta': {'content': '[]}'}, 'finish_reason': 'stop'}]}]
        response = io.BytesIO((''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n').encode())
        self.assertEqual(self.planner(events)._read_stream(response), '{"route":[]}')
        self.assertEqual(events[0]['field'], 'reasoning_content')
        self.assertEqual(events[0]['text'], '公开的推理文本')

    def test_truncated_or_length_limited_stream_rejected(self):
        for ending in ('', 'data: [DONE]\n\n',
                       'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'):
            response = io.BytesIO(('data: {"choices":[{"delta":{"content":"{}"}}]}\n\n'+ending).encode())
            with self.subTest(ending=ending), self.assertRaises(RuntimeError):
                self.planner([])._read_stream(response)

    def test_timeout_has_stage_and_completion_event(self):
        events = []
        with patch('planning.request.urlopen', side_effect=TimeoutError('timeout')):
            with self.assertRaisesRegex(RuntimeError, '不是机器人行走超时'):
                self.planner(events)._request('system', {'scene': {}})
        self.assertEqual(events[-1]['state'], 'model_timeout')

    def test_live_wait_and_bounded_trace(self):
        app = LiveApp.__new__(LiveApp)
        app.lock = threading.RLock()
        app.state = {'active_job': 1}
        app.emit({'type': 'model_request', 'state': 'started', 'stage': '路线规划'})
        self.assertLessEqual(app.model_started, time.monotonic())
        for i in range(170):
            app.emit({'type': 'segment', 'status': 'succeeded', 'target_node': str(i)})
        self.assertEqual(len(app.state['execution_trace']), 150)


if __name__ == '__main__':
    unittest.main()
