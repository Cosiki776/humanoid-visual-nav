"""Exercise the real browser status formatter with minimal DOM stubs."""
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('node'), 'node required')
class CallStatusUITests(unittest.TestCase):
    def test_detect_wait_return_block_and_move(self):
        source = Path(__file__).with_name('visual_web.html').read_text()
        function = source.split('function renderCallStatus(s){', 1)[1].split('\nconst phases=', 1)[0]
        script = "const callStatus={},actions={select:'选择目标'};\nfunction renderCallStatus(s){"+function
        script += r'''
const assert=require('assert');
const obs={type:'observation',id:'o1',seen:['v1']};
const waiting={type:'vlm_call',call:1,status:'waiting',observation_id:'o1',image_count:2,timestamp:1};
const returned={...waiting,status:'returned',timestamp:2,answer:{candidate_id:'v1',category_match:'yes',attributes_match:'yes'}};
renderCallStatus({vlm_observability:true,events:[obs]});
assert(callStatus.textContent.includes('尚未送模型'));
renderCallStatus({busy:true,events:[obs],vlm_status:waiting});
assert(callStatus.textContent.includes('等待返回'));
assert(callStatus.textContent.includes('已送模型'));
renderCallStatus({events:[obs,waiting,returned,{type:'approach_blocked',candidate_id:'v1',reason:'no dock'}]});
assert(callStatus.textContent.includes('建议靠近 v1'));
assert(callStatus.textContent.includes('未能继续靠近：no dock'));
renderCallStatus({events:[obs,waiting,returned,{type:'target_motion',candidate_id:'v1'}]});
assert(callStatus.textContent.includes('已下发靠近指令'));
renderCallStatus({events:[obs,waiting,returned,{...obs,id:'o2'}]});
assert(callStatus.textContent.includes('o2 · 1 个候选 · 尚未送模型'));
renderCallStatus({events:[obs]});
assert(callStatus.textContent.includes('旧会话无法判断'));
'''
        subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
