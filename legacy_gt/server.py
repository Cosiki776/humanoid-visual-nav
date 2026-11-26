#!/usr/bin/env python3
"""Local browser UI + MJPEG stream + persistent, serial Habitat agent session."""
import copy
import io
import json
import queue
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import bootstrap
from app.model_settings import load_model_settings


class LiveApp:
    def __init__(self, scene="apt_0", radius=0.3, realtime=True):
        model_config = load_model_settings()
        self.scene_name, self.radius, self.realtime = scene, radius, realtime
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.shutdown = threading.Event()
        self.cancel = threading.Event()
        self.pause_requested = threading.Event()
        self.commands = queue.Queue(maxsize=8)
        self.jpeg, self.frame_id = None, 0
        self.counter = 0
        self.log_dir = bootstrap.ROOT / "outputs" / ("session_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        self.log_dir.mkdir(parents=True)
        self.state = {"phase": "loading", "ready": False, "active_job": None, "engine": "langgraph",
                      "scene": scene, "position": None, "targets": [], "messages": [], "jobs": [],
                      "model_configured": model_config["configured"],
                      "model": model_config["model"], "error": None}

    def snapshot(self):
        with self.lock:
            state = dict(copy.deepcopy(self.state), frame_id=self.frame_id, queued=self.commands.qsize())
            pending = state.get("model_request")
            if pending and pending["state"] == "started":
                pending["elapsed_seconds"] = round(time.monotonic()-self.model_started, 1)
            return state

    def submit(self, data):
        if not isinstance(data, dict):
            raise ValueError("请求必须是 JSON 对象")
        instruction = data.get("instruction", "")
        target = data.get("target")
        targets = data.get("targets")
        replace = data.get("replace", False)
        return_to_start = data.get("return_to_start", False)
        if not isinstance(replace, bool) or not isinstance(return_to_start, bool):
            raise ValueError("replace 和 return_to_start 必须是布尔值")
        if targets is not None and (not isinstance(targets, list) or not 1 <= len(targets) <= 7
                                    or any(not isinstance(t, str) for t in targets)):
            raise ValueError("离线多目标列表需要 1 到 7 个目标 ID")
        if target is not None and targets is not None:
            raise ValueError("不能同时指定 target 和 targets")
        if not isinstance(instruction, str) or len(instruction) > 4000:
            raise ValueError("指令必须是 4000 字符以内的文本")
        if target is not None and not isinstance(target, str):
            raise ValueError("目标 ID 无效")
        instruction = instruction.strip()
        with self.lock:
            if not self.state["ready"]:
                raise ValueError("场景尚未就绪")
            if target is None and targets is None:
                control = instruction.strip(" 。！!，,")
                if control in {"暂停", "暂停一下", "先暂停"}:
                    self.pause()
                    return {"control": "paused", "message": "已请求暂停"}
                if control in {"继续", "继续执行", "恢复"}:
                    self.resume()
                    return {"control": "resumed", "message": "已请求继续"}
                if control in {"停止", "停下", "取消"}:
                    self.stop()
                    return {"control": "cancelled", "message": "已请求停止"}
                for prefix in ("改成", "换成", "改去", "换去"):
                    if instruction.startswith(prefix):
                        replace = True
                        instruction = instruction[len(prefix):].strip()
                        break
            steps = None
            if targets is not None:
                available = {o["id"] for o in self.state["targets"]}
                if any(t not in available for t in targets):
                    raise ValueError("列表中存在不可达或未知目标")
                steps = [{"action": "navigate", "target": t} for t in targets]
                if return_to_start:
                    steps.append({"action": "return", "location": "task_start"})
                instruction = "离线多步骤测试：" + " → ".join(targets) + (" → 返回任务起点" if return_to_start else "")
            if target is not None:
                if target not in {o["id"] for o in self.state["targets"]}:
                    raise ValueError("目标不存在或没有可达站立点")
                instruction = "离线目标测试：走到 " + target + " 旁边"
            elif not instruction:
                raise ValueError("请输入指令")
            elif targets is None and not self.state["model_configured"]:
                raise ValueError("模型尚未配置，请填写 config/model_config.json 后重启服务，或使用离线目标测试。")
            if replace:
                self.stop()
            if self.commands.full():
                raise ValueError("等待队列已满，请等待当前任务完成")
            self.counter += 1
            job = {"id": self.counter, "instruction": instruction, "target": target, "steps": steps, "status": "queued"}
            self.state["jobs"].append(copy.deepcopy(job))
            self.state["jobs"] = self.state["jobs"][-100:]
            self.commands.put_nowait(job)
            self.emit({"type": "lifecycle", "job": job["id"], "message": "已接收指令并加入队列"})
            return job

    def stop(self):
        with self.lock:
            self.cancel.set()
            self.pause_requested.clear()
            while True:
                try:
                    dropped = self.commands.get_nowait()
                    for job in self.state["jobs"]:
                        if job["id"] == dropped["id"]:
                            job["status"] = "cancelled"
                    self.commands.task_done()
                except queue.Empty:
                    break

    def pause(self):
        with self.lock:
            if self.state["active_job"] is None:
                raise ValueError("当前没有正在执行的任务")
            self.pause_requested.set()

    def resume(self):
        self.pause_requested.clear()

    def control_gate(self):
        if self.cancel.is_set() or self.shutdown.is_set():
            return True
        if self.pause_requested.is_set():
            with self.lock:
                previous = self.state["phase"]
                self.state["phase"] = "paused"
            self.executor.stop()
            while self.pause_requested.is_set() and not self.cancel.is_set() and not self.shutdown.is_set():
                self.shutdown.wait(0.05)
            with self.lock:
                self.state["phase"] = previous
        return self.cancel.is_set() or self.shutdown.is_set()

    def publish_frame(self, frame):
        from PIL import Image
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="JPEG", quality=82)
        with self.frame_ready:
            self.jpeg = buffer.getvalue()
            self.frame_id += 1
            if getattr(self, "executor", None) is not None:
                self.state["position"] = [float(v) for v in self.executor.position]
            self.frame_ready.notify_all()

    def emit(self, event):
        with self.lock:
            kind = event["type"]
            if kind == "model_delta":
                field = event["field"]
                stream = self.state.setdefault("model_stream", {"reasoning_content": "", "content": ""})
                stream[field] = (stream.get(field, "") + event["text"])[-60000:]
                return
            if kind == "model_request":
                self.state["model_request"] = copy.deepcopy(event)
                if event["state"] == "started":
                    self.model_started = time.monotonic()
                    self.state["model_stream"] = {"reasoning_content": "", "content": ""}
            if kind in {"lifecycle", "phase", "provider_response", "model_request", "task_progress", "plan", "validation", "segment", "result"}:
                # Only observable tool events; never API headers or model reasoning tokens.
                record = {"time": datetime.now().strftime("%H:%M:%S"), "job": self.state["active_job"],
                          **copy.deepcopy(event)}
                if kind == "result":
                    record["result"] = {k: event["result"].get(k) for k in
                                        ("status", "message", "completed_steps")}
                self.state.setdefault("execution_trace", []).append(record)
                self.state["execution_trace"] = self.state["execution_trace"][-150:]
                if hasattr(self, "log_dir"):
                    with (self.log_dir / "execution_trace.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print("TRACE " + json.dumps(record, ensure_ascii=False), flush=True)
            if event["type"] == "message":
                self.state["messages"].append({"role": event["role"], "content": event["content"]})
                self.state["messages"] = self.state["messages"][-60:]
            elif event["type"] == "phase":
                self.state["phase"] = event["phase"]
            elif event["type"] == "plan":
                self.state["plan"] = event["plan"]
                self.state["route_view"] = {k: event.get(k) for k in ("source", "route_points")}
            elif event["type"] == "task_progress":
                self.state["task_progress"] = event
            elif event["type"] == "result":
                self.state["last_result"] = event["result"]

    def worker(self):
        # All GL, simulator, and controller calls stay on this owning thread.
        from simulation.world import World
        from simulation.executor import Executor
        from legacy_gt.planning import ModelPlanner
        from legacy_gt.session import NavigationSession
        from legacy_gt.task_graph import TaskWorkflow
        world = None
        try:
            world = World(self.scene_name, self.radius)
            attrs = json.loads((bootstrap.ROOT / "legacy_gt/annotations.json").read_text())
            scene = world.build_map(attrs)
            start = min(scene["nodes"])
            self.executor = Executor(world, scene["nodes"][start], self.publish_frame,
                                     realtime=self.realtime, trace_limit=18000)
            planner = ModelPlanner(emit=self.emit) if self.state["model_configured"] else None
            session = NavigationSession(world, self.executor, scene, planner, self.emit)
            workflow = TaskWorkflow(session)
            self.publish_frame(world.frame(self.executor.position))
            (self.log_dir / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2))
            with self.lock:
                self.state.update(ready=True, phase="idle", targets=[o for o in scene["objects"] if o["goal_nodes"]],
                                  navigation_map={"nodes": copy.deepcopy(scene["nodes"]),
                                                  "edges": copy.deepcopy(scene["edges"])})
            print(f"SESSION READY: {self.log_dir}", flush=True)
            while not self.shutdown.is_set():
                # Dequeue and reset cancellation atomically with stop() to avoid losing a stop.
                with self.lock:
                    try:
                        job = self.commands.get_nowait()
                    except queue.Empty:
                        job = None
                    if job is not None:
                        self.cancel.clear()
                        self.pause_requested.clear()
                        self.state.update(active_job=job["id"], phase="planning_tasks", plan=None, task_progress=None,
                                          model_request=None, route_view=None, model_stream=None)
                        for item in self.state["jobs"]:
                            if item["id"] == job["id"]:
                                item["status"] = "running"
                if job is None:
                    self.shutdown.wait(0.1)
                    continue
                result = workflow.run_turn(job["instruction"], job["target"], self.control_gate, steps=job["steps"])
                self.executor.stop()
                self.publish_frame(world.frame(self.executor.position))
                (self.log_dir / f"turn_{job['id']:04d}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                with self.lock:
                    for item in self.state["jobs"]:
                        if item["id"] == job["id"]:
                            item.update(status=result["status"], result=result)
                    self.state.update(active_job=None, phase="idle")
                self.commands.task_done()
                print(f"TURN {job['id']}: {result['status']}", flush=True)
        except Exception as exc:
            with self.lock:
                self.state.update(phase="error", ready=False, error=f"{type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            if world is not None:
                world.close()


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_body(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json_response(self, code, data):
            self.send_body(code, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_body(200, (bootstrap.ROOT / "legacy_gt/web.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self.json_response(200, app.snapshot())
            elif path == "/api/frame.jpg":
                with app.lock:
                    jpeg = app.jpeg
                if jpeg is None:
                    self.json_response(503, {"error": "画面尚未就绪"})
                else:
                    self.send_body(200, jpeg, "image/jpeg")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last_id = -1
                self.connection.settimeout(5)
                try:
                    while not app.shutdown.is_set():
                        with app.frame_ready:
                            app.frame_ready.wait_for(lambda: app.frame_id != last_id or app.shutdown.is_set(), timeout=1)
                            jpeg, last_id = app.jpeg, app.frame_id
                        if jpeg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                             str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                except (OSError, TimeoutError):
                    pass
            else:
                self.json_response(404, {"error": "Not found"})

        def do_POST(self):
            # Local single-user service; JSON + same-origin prevents cross-site form submissions.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self.json_response(403, {"error": "Cross-origin request rejected"})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.json_response(415, {"error": "Expected application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("请求大小无效")
                data = json.loads(self.rfile.read(length))
                if self.path == "/api/command":
                    self.json_response(202, app.submit(data))
                elif self.path == "/api/stop":
                    app.stop()
                    self.json_response(200, {"status": "stop_requested"})
                elif self.path == "/api/pause":
                    app.pause()
                    self.json_response(200, {"status": "pause_requested"})
                elif self.path == "/api/resume":
                    app.resume()
                    self.json_response(200, {"status": "resume_requested"})
                else:
                    self.json_response(404, {"error": "Not found"})
            except (ValueError, UnicodeDecodeError) as exc:
                self.json_response(400, {"error": str(exc)})
    return Handler


