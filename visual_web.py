"""Visual Web transport: persistent visual sessions, visual evidence only, explicit run history."""
import copy
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import bootstrap
from model_settings import load_model_settings


class VisualApp:
    def __init__(self, scene='apt_0'):
        self.scene = scene
        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.process = None
        self.jpeg = None
        self.frame_id = 0
        self.busy = False
        self.cancelled = False
        self.run_dir = None
        self.started = None
        self.session_id = 'visual_session_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.session_state = {}
        self.session_jpeg = None
        self.task_count = 0
        self.state = dict(phase='idle', mode='visual', ready=True, events=[], memory={}, observations=0)
        config = load_model_settings()
        self.model = os.environ.get('NAV_VLM_MODEL', config['model'])
        self.configured = bool(self.model and (os.environ.get('NAV_VLM_BASE_URL') or config['base_url']))

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.state)
            result.update(busy=self.busy, scene=self.scene, model=self.model, active_session_id=self.session_id, task_count=self.task_count,
                          configured=self.configured, observer_frame_id=self.frame_id, observer_available=self.jpeg is not None, run=self.run_dir.name if self.run_dir else None)
            if self.busy and self.started:
                result['wall_seconds'] = round(time.monotonic()-self.started, 1)
            return result

    def runs(self):
        rows = []
        for path in sorted((bootstrap.ROOT/'outputs').glob('*/result.json'), key=lambda p:p.stat().st_mtime, reverse=True):
            try:
                r = json.loads(path.read_text())
                if r.get('policy') not in {'VLMPolicy', 'ColorPolicy'} or r.get('mode') == 'gt':
                    continue
                rows.append(dict(id=path.parent.name, instruction=r.get('instruction',''), status=r['status'],
                                 observations=r.get('observations',0)))
            except (ValueError, OSError, KeyError):
                continue
            if len(rows) == 30:
                break
        return rows

    def load(self, run):
        with self.lock:
            if self.busy:
                raise ValueError('请先停止当前任务，再查看历史记录')
            if not isinstance(run, str) or run not in {r['id'] for r in self.runs()}:
                raise ValueError('未找到视觉运行记录')
            self.run_dir = bootstrap.ROOT/'outputs'/run
            r = json.loads((self.run_dir/'result.json').read_text())
            r.pop('selected_points', None)
            self.state = dict(r, phase=r['status'], replay=True, mode='visual', ready=True)
            observer = self.run_dir/'observer_last.jpg'
            self.jpeg = observer.read_bytes() if observer.exists() else None
            self.frame_id += 1
            # Summary records contain evidence per instance, not observation history.
            frames = sorted(self.run_dir.glob('obs_*.jpg'))
            frames = [p for p in frames if re.fullmatch(r'obs_\d+\.jpg',p.name)]
            if frames:
                self.state['observation'] = dict(id=frames[-1].stem, rgb_path=str(frames[-1]))
            return self.snapshot()

    def initialize(self):
        """Warm up the simulator once, without creating a navigation task."""
        with self.lock:
            if self.busy or self.process is not None or self.jpeg is not None or self.state.get('phase') == 'error':
                return self.snapshot()
            session_dir = bootstrap.ROOT/'outputs'/self.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            self.busy = True
            self.started = time.monotonic()
            self.state.update(phase='loading', instruction='加载初始画面')
            threading.Thread(target=self.worker, args=(None, 0, session_dir/'preview'), daemon=True).start()
            return self.snapshot()

    def submit(self, data):
        instruction = data.get('instruction','')
        budget = data.get('max_observations',24)
        if not isinstance(instruction,str) or not 1 <= len(instruction.strip()) <= 4000:
            raise ValueError('请输入 1 到 4000 字的导航指令')
        if isinstance(budget,bool) or not isinstance(budget,int) or not 1 <= budget <= 200:
            raise ValueError('观察预算必须为 1 到 200')
        with self.lock:
            if self.busy:
                raise ValueError('当前任务正在执行，请先停止或等待完成')
            from visual_nav.directional import parse_directional
            if not self.configured and parse_directional(instruction) is None:
                raise ValueError('请先配置视觉模型 API 后重启服务')
            if self.process is not None and self.process.poll() is not None:
                raise ValueError('会话进程已退出，当前位置无法继续；请新建会话，不会自动回到起点')
            self.busy, self.cancelled = True, False
            self.jpeg = self.session_jpeg
            self.frame_id += 1
            self.started = time.monotonic()
            self.run_dir = bootstrap.ROOT/'outputs'/('web_visual_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            session_dir = bootstrap.ROOT/'outputs'/self.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir/'cancel.request').unlink(missing_ok=True)
            self.state = dict(phase='loading', mode='visual', replay=False, ready=True,
                              session_id=self.session_id, position=self.session_state.get('final_position', self.session_state.get('position')),
                              instruction=instruction.strip(), events=[], memory=self.session_state.get('memory', {}), observations=0, budget=budget)
            thread = threading.Thread(target=self.worker, args=(instruction.strip(),budget,self.run_dir), daemon=True)
            thread.start()
            return dict(status='accepted', run=self.run_dir.name)

    def stop(self):
        with self.lock:
            if self.busy:
                self.cancelled = True
                self.state['phase'] = 'stopping'
                (bootstrap.ROOT/'outputs'/self.session_id/'cancel.request').touch()
            if self.shutdown.is_set() and self.process and self.process.poll() is None:
                self.process.terminate()

    def new_session(self):
        with self.lock:
            if self.busy:
                raise ValueError('请先停止当前任务，待机器人停止后再新建会话')
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.process:
                for pipe in (self.process.stdin, self.process.stdout):
                    if pipe:
                        pipe.close()
            self.process = None
            self.session_id = 'visual_session_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.session_state, self.session_jpeg = {}, None
            self.task_count = 0
            self.jpeg, self.run_dir = None, None
            self.frame_id += 1
            self.state = dict(phase='idle', mode='visual', replay=False, ready=True,
                              events=[], memory={}, observations=0)
            return self.snapshot()

    def worker(self, instruction, budget, output):
        try:
            log = output.parent/(output.name+'.log')
            with log.open('w') as logfile:
                with self.lock:
                    if self.process is None:
                        session_dir = bootstrap.ROOT/'outputs'/self.session_id
                        command = [sys.executable, '-u', str(bootstrap.ROOT/'run_visual_session.py'),
                                   '--scene', self.scene, '--vlm-model', self.model,
                                   '--session-dir', str(session_dir)]
                        with (session_dir/'worker.log').open('a') as errors:
                            self.process = subprocess.Popen(command, cwd=bootstrap.ROOT,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                text=True, bufsize=1)
                    process = self.process
                    process.stdin.write(json.dumps(dict(instruction=instruction,
                        max_observations=budget, output=str(output)), ensure_ascii=False)+'\n')
                    process.stdin.flush()
                completed = False
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        logfile.write(line)
                        logfile.flush()
                        continue
                    if event.get('type') == 'observer_frame':
                        with self.lock:
                            self.jpeg = base64.b64decode(event['jpeg'])
                            self.session_jpeg = self.jpeg
                            self.frame_id += 1
                        continue
                    logfile.write(line)
                    logfile.flush()
                    with self.lock:
                        if event.get('type') == 'live_state':
                            self.state.update({k:v for k,v in event.items() if k!='type'})
                            if self.cancelled:
                                self.state['phase'] = 'stopping'
                        elif event.get('type') == 'run_error':
                            self.state.update(phase='error', error=event.get('message'), error_type=event.get('error_type'))
                        elif event.get('type') == 'task':
                            self.state.update(task=event['task'], phase='perception',
                                              instruction_parse_seconds=event.get('parse_seconds'))
                        elif event.get('type') == 'task_done':
                            if event.get('run') != output.name:
                                raise RuntimeError('Session task response mismatch')
                            completed = True
                            break
                        elif event.get('type') == 'session_ready' and instruction is None:
                            self.state.update(phase='idle', instruction='', position=event.get('position'))
                            completed = True
                            break
                with self.lock:
                    if completed and instruction is None:
                        pass
                    elif completed and (output/'result.json').exists():
                        result = json.loads((output/'result.json').read_text())
                        result.pop('selected_points',None)
                        self.state.update(result, phase=result['status'])
                        self.task_count += 1
                    else:
                        self.state.update(phase='error', error='会话进程异常退出，请检查会话 worker.log 并新建会话')
        except Exception as exc:
            with self.lock:
                self.state.update(phase='error',error=type(exc).__name__+': '+str(exc))
        finally:
            with self.lock:
                self.busy = False
                self.state['wall_seconds'] = round(time.monotonic()-self.started,1)
                self.session_state = copy.deepcopy(self.state)

    def asset(self, run, name):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',run) or not re.fullmatch(r'(?:observer_last|obs_\d+(?:_\d+_(?:crop|mask))?)\.(?:jpg|png)',name):
            raise ValueError('无效图像路径')
        root = (bootstrap.ROOT/'outputs').resolve()
        path = (root/run/name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError('图像不存在')
        return path


def handler_for_visual(app, legacy_factory):
    from serve import handler_for
    from http.server import BaseHTTPRequestHandler
    legacy = {'app':None,'handler':None}
    legacy_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def body(self,code,data,kind='application/json; charset=utf-8'):
            if not isinstance(data,bytes):
                data=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):
                pass

        def gt(self, method):
            with legacy_lock:
                if legacy['app'] is None:
                    legacy['app']=legacy_factory()
                    legacy['handler']=handler_for(legacy['app'])
                    threading.Thread(target=legacy['app'].worker,daemon=True).start()
            self.path=self.path[3:] or '/'
            if method=='GET' and self.path=='/':
                html=(bootstrap.ROOT/'web.html').read_text().replace('/api/','/gt/api/').replace('/stream.mjpg','/gt/stream.mjpg')
                html=html.replace('<main>','<p><a href="/" style="color:#67dfb7">← 返回视觉导航</a> · GT 基线：此页面使用真实物体表</p><main>')
                self.body(200,html.encode(),'text/html; charset=utf-8')
            else:
                # Delegate legacy behavior with its original closure and methods.
                delegate=legacy['handler']
                for name in ('send_body','json_response'):
                    setattr(self,name,getattr(delegate,name).__get__(self,type(self)))
                getattr(delegate,'do_'+method)(self)

        def do_GET(self):
            parsed=urlparse(self.path)
            try:
                if parsed.path=='/gt' or parsed.path.startswith('/gt/'):
                    return self.gt('GET')
                if parsed.path=='/':
                    return self.body(200,(bootstrap.ROOT/'visual_web.html').read_bytes(),'text/html; charset=utf-8')
                if parsed.path=='/api/observer.jpg':
                    with app.lock:
                        jpeg=app.jpeg
                    return self.body(200,jpeg,'image/jpeg') if jpeg else self.body(503,dict(error='等待第三人称画面'))
                if parsed.path=='/observer.mjpg':
                    self.send_response(200)
                    self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control','no-store')
                    self.end_headers()
                    self.connection.settimeout(5)
                    last=-1
                    last_sent=0.0
                    try:
                        self.wfile.write(b'--frame\r\n')
                        while not app.shutdown.is_set():
                            with app.lock:
                                jpeg,frame_id=app.jpeg,app.frame_id
                            now=time.monotonic()
                            if jpeg and (frame_id!=last or now-last_sent>=.5):
                                # Close each multipart image immediately. Idle heartbeat repeats
                                # the actual cached frame; it does not advance the simulator.
                                self.wfile.write(b'Content-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n--frame\r\n')
                                self.wfile.flush()
                                last=frame_id
                                last_sent=now
                            app.shutdown.wait(.04)
                    except (OSError,TimeoutError):
                        pass
                    return
                if parsed.path=='/api/status':
                    return self.body(200,app.snapshot())
                if parsed.path=='/api/runs':
                    return self.body(200,app.runs())
                if parsed.path=='/api/asset':
                    q=parse_qs(parsed.query)
                    p=app.asset(q.get('run',[''])[0],q.get('name',[''])[0])
                    return self.body(200,p.read_bytes(),'image/png' if p.suffix=='.png' else 'image/jpeg')
                return self.body(404,dict(error='Not found'))
            except (ValueError,OSError) as exc:
                self.body(400,dict(error=str(exc)))

        def do_POST(self):
            if self.path.startswith('/gt/'):
                return self.gt('POST')
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc!=self.headers.get('Host'):
                return self.body(403,dict(error='Cross-origin request rejected'))
            try:
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    raise ValueError('Expected application/json')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:
                    raise ValueError('请求大小无效')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):
                    raise ValueError('Expected JSON object')
                if self.path=='/api/command':
                    return self.body(202,app.submit(data))
                if self.path=='/api/session/initialize':
                    return self.body(200,app.initialize())
                if self.path=='/api/load':
                    return self.body(200,app.load(data.get('run')))
                if self.path=='/api/session/new':
                    return self.body(200,app.new_session())
                if self.path=='/api/stop':
                    app.stop()
                    return self.body(200,dict(status='stop_requested'))
                self.body(404,dict(error='Not found'))
            except (ValueError,OSError) as exc:
                self.body(400,dict(error=str(exc)))
    Handler.legacy_apps=legacy
    return Handler
