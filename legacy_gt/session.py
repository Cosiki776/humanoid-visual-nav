"""One world and one humanoid for a sequence of conversational tasks."""
import copy
import numpy as np
from legacy_gt.planning import smoke_plan, validate_plan


class NavigationSession:
    def __init__(self, world, executor, scene, planner=None, emit=None, max_replans=2):
        self.world = world
        self.executor = executor
        self.scene = scene
        self.planner = planner
        self.emit = emit or (lambda event: None)
        self.max_replans = max_replans
        self.history = []
        self.turn = 0

    def remember(self, role, content, publish=True):
        self.history.append({"role": role, "content": content})
        self.history = self.history[-40:]
        if publish:
            self.emit({"type": "message", "role": role, "content": content})

    def current_map(self, disabled_edges):
        # Copy the persistent map: temporary start points never accumulate across turns.
        scene = copy.deepcopy(self.scene)
        current = "session_current"
        position = np.asarray(self.executor.position, dtype=float)
        scene["nodes"][current] = position.tolist()
        scene["edges"] = [e for e in scene["edges"] if frozenset(e) not in disabled_edges]
        for node, point in self.scene["nodes"].items():
            if (frozenset((current, node)) not in disabled_edges
                    and np.linalg.norm(position-point) <= 1.65 and self.world.direct(position, point)):
                scene["edges"].append([current, node])
        # An already reached goal is also valid without moving away and back.
        for obj in scene["objects"]:
            if any(np.linalg.norm(position-scene["nodes"][g]) < 0.08 for g in obj["goal_nodes"]):
                obj["goal_nodes"].append(current)
        scene["current_node"] = current
        return scene, current

    def run_turn(self, instruction, target=None, should_stop=lambda: False, internal=False, required_target=None):
        self.turn += 1
        prior_history = copy.deepcopy(self.history)
        if not internal:
            self.remember("user", instruction)
        start_position = np.asarray(self.executor.position).tolist()
        result = {"turn": self.turn, "instruction": instruction,
                  "mode": "offline_target" if target else "agent",
                  "start_position": start_position, "status": "invalid_plan", "events": []}
        feedback, disabled = None, set()
        try:
            for attempt in range(self.max_replans+1):
                if should_stop():
                    result["status"] = "cancelled"
                    break
                scene, start = self.current_map(disabled)
                self.emit({"type": "phase", "phase": "planning", "attempt": attempt})
                try:
                    if target:
                        plan = smoke_plan(scene, start, target)
                    elif self.planner is None:
                        raise RuntimeError("未配置模型。请填写 humanoid_nav_demo/config/model_config.json 后重启服务，或使用离线目标测试。")
                    else:
                        plan = self.planner.plan(instruction, scene, start, feedback, history=prior_history)
                    if should_stop():
                        result["status"] = "cancelled"
                        break
                    if isinstance(plan, dict) and plan.get("status") in {"needs_clarification", "unsupported"}:
                        result.update(status=plan["status"], message=str(plan.get("message", "请补充目标信息。")))
                        break
                    validate_plan(plan, scene, start)
                    if required_target and plan["target_object"] != required_target:
                        raise ValueError("Route must end at the requested saved location")
                except (ValueError, KeyError, TypeError, StopIteration) as exc:
                    feedback = {"status": "invalid_plan", "reason": str(exc)}
                    result["events"].append(feedback)
                    self.emit({"type": "validation", "attempt": attempt, **feedback})
                    if target:
                        break
                    continue
                result["target_object"] = plan["target_object"]
                result["events"].append({"plan": plan, "attempt": attempt})
                self.emit({"type": "plan", "plan": plan,
                           "source": "offline_bfs" if target else "model",
                           "route_points": [scene["nodes"][node] for node in plan["route"]]})
                blocked = False
                for previous, node in zip(plan["route"], plan["route"][1:]):
                    self.emit({"type": "phase", "phase": "walking", "target_node": node})
                    event = self.executor.execute_segment(scene["nodes"][node], should_stop=should_stop)
                    result["events"].append(dict(event, target_node=node))
                    self.emit({"type": "segment", **event, "target_node": node})
                    if event["status"] != "succeeded":
                        blocked = True
                        disabled.add(frozenset((previous, node)))
                        feedback = event
                        result["status"] = event["status"]
                        break
                if should_stop():
                    result["status"] = "cancelled"
                    break
                if not blocked:
                    distance = float(np.linalg.norm(np.asarray(self.executor.position) - scene["nodes"][plan["route"][-1]]))
                    result.update(status="succeeded" if distance < 0.1 else "failed_arrival", final_error_m=distance)
                    break
                if target:
                    break
        except Exception as exc:
            # A failed turn must not destroy the scene or terminate subsequent turns.
            result.update(status="error", message=str(exc))
        result["final_position"] = np.asarray(self.executor.position).tolist()
        messages = {"succeeded": "已到达目标旁边，可以继续输入下一条指令。",
                    "cancelled": "当前任务已停止，下一条指令将从当前位置开始。",
                    "blocked": "行走受阻，请调整目标后继续。",
                    "timeout": "执行超时，请调整目标后继续。",
                    "invalid_plan": "未能生成有效路线，请补充或调整指令。"}
        message = result.get("message") or messages.get(result["status"], "本次任务未完成，可以继续发送指令。")
        if internal and result["status"] == "succeeded":
            message = "当前导航步骤已完成。"
        # The model receives outcome and target identity, not just the user's old commands.
        summary = {"status": result["status"], "message": message,
                   "target_object": result.get("target_object"), "position": result["final_position"]}
        self.remember("assistant", summary, publish=not internal)
        if not internal:
            self.emit({"type": "result", "result": result})
        return result
