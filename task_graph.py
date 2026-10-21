"""LangGraph task orchestration. GL actions run outside graph worker threads."""
import copy
import re
import threading
from typing import Any, Dict, List, TypedDict

import bootstrap
import numpy as np
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class TaskState(TypedDict, total=False):
    steps: List[Dict[str, Any]]
    index: int
    result: Dict[str, Any]
    completed: int


def validate_tasks(plan, offline=False):
    if not isinstance(plan, dict):
        raise ValueError("任务计划必须是 JSON 对象")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        raise ValueError("任务必须包含 1 到 8 个步骤")
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("任务步骤必须是对象")
        if step.get("action") == "navigate":
            if offline:
                if not isinstance(step.get("target"), str) or not step["target"]:
                    raise ValueError("离线任务需要目标 ID")
            elif not isinstance(step.get("instruction"), str) or not 1 <= len(step["instruction"]) <= 4000:
                raise ValueError("导航步骤需要单目标指令")
        elif step.get("action") == "return":
            if step.get("location") not in {"task_start", "session_start"}:
                raise ValueError("返回位置只能是 task_start 或 session_start")
        else:
            raise ValueError("当前只支持 navigate 和 return")
    return steps


class TaskWorkflow:
    def __init__(self, session):
        self.session = session
        self.session_start = np.asarray(session.executor.position).tolist()
        self.task_number = 0
        self.owner = threading.get_ident()

    def save_anchor(self, name, position):
        scene, world = self.session.scene, self.session.world
        node = "saved_" + name
        scene["nodes"][node] = list(position)
        scene["edges"] = [e for e in scene["edges"] if node not in e]
        for key, point in scene["nodes"].items():
            if key != node and np.linalg.norm(np.asarray(position)-point) <= 1.65 and world.direct(position, point):
                scene["edges"].append([node, key])
        scene["objects"] = [o for o in scene["objects"] if o["id"] != node]
        scene["objects"].append({"id": node, "category": "saved_location", "position_world": list(position),
                                 "attributes": {"name": name}, "goal_nodes": [node]})
        return node

    def graph(self):
        builder = StateGraph(TaskState)
        builder.add_node("select_step", lambda state: {})
        # An interrupt boundary, never automatically executed by LangGraph.
        def external_action(state):
            raise RuntimeError("Habitat actions must execute on the simulator owner thread")
        builder.add_node("execute", external_action)
        builder.add_node("advance", lambda state: {"index": state["index"]+1, "completed": state["completed"]+1})
        builder.add_edge(START, "select_step")
        builder.add_conditional_edges("select_step", lambda s: "execute" if s["index"] < len(s["steps"]) else END)
        builder.add_conditional_edges("execute", lambda s: "advance" if s["result"]["status"] == "succeeded" else END)
        builder.add_edge("advance", "select_step")
        return builder.compile(checkpointer=MemorySaver(), interrupt_before=["execute"])

    def run_turn(self, instruction, target=None, should_stop=lambda: False, steps=None):
        if threading.get_ident() != self.owner:
            raise RuntimeError("TaskWorkflow must run on its owning simulator thread")
        session = self.session
        self.task_number += 1
        origin = np.asarray(session.executor.position).tolist()
        history = copy.deepcopy(session.history)
        session.remember("user", instruction)
        result = {"turn": self.task_number, "instruction": instruction, "engine": "langgraph",
                  "mode": "offline_target" if target or steps is not None else "agent",
                  "start_position": origin, "status": "error", "events": [], "completed_steps": 0}
        try:
            session.emit({"type": "phase", "phase": "planning_tasks"})
            if target:
                steps = [{"action": "navigate", "target": target}]
            if steps is None:
                if session.planner is None:
                    raise ValueError("请先配置模型，或使用离线目标测试")
                # Fast rejection of explicit sequencing; semantic checking also belongs
                # to the single route-model call. No separate decomposition request.
                if re.search(r"然后|接着|随后|最后|再去|再到|依次|分别|转三圈|then\b|afterwards\b", instruction, re.I):
                    plan = {"status": "unsupported", "message": "自然语言暂时只支持一个导航目标，请拆成多条指令发送。离线多目标测试仍保留。"}
                else:
                    plan = {"steps": [{"action": "navigate", "instruction": instruction}]}
            else:
                plan = {"steps": steps}
            if should_stop():
                result["status"] = "cancelled"
            elif isinstance(plan, dict) and plan.get("status") in {"needs_clarification", "unsupported"}:
                result.update(status=plan["status"], message=str(plan.get("message", "请调整任务。")))
            else:
                task_steps = validate_tasks(plan, offline=result["mode"] == "offline_target")
                result["steps"] = task_steps
                self.save_anchor("session_start", self.session_start)
                self.save_anchor("task_start", origin)
                graph = self.graph()
                config = {"configurable": {"thread_id": str(self.task_number)}, "recursion_limit": 100}
                graph.invoke({"steps": task_steps, "index": 0, "completed": 0}, config)
                while graph.get_state(config).next:
                    state = graph.get_state(config).values
                    index = state["index"]
                    session.emit({"type": "task_progress", "steps": task_steps, "index": index,
                                  "completed": state["completed"], "engine": "langgraph"})
                    if should_stop():
                        result["status"] = "cancelled"
                        break
                    step = task_steps[index]
                    offline_target = step.get("target") if result["mode"] == "offline_target" else None
                    required = None
                    if step["action"] == "return":
                        required = "saved_" + step["location"]
                        text = "返回保存位置，对象 ID 为 " + required
                        if result["mode"] == "offline_target":
                            offline_target = required
                    else:
                        text = step.get("instruction", "走到 " + str(offline_target) + " 旁边")
                    # This action intentionally occurs on the caller/GL owner, NOT inside a graph node.
                    step_result = session.run_turn(text, offline_target, should_stop, internal=True, required_target=required)
                    result["events"].append({"step_index": index, "result": step_result})
                    graph.update_state(config, {"result": step_result}, as_node="execute")
                    graph.invoke(None, config)
                    snapshot = graph.get_state(config).values
                    result["completed_steps"] = snapshot["completed"]
                    result["status"] = step_result["status"]
                    if step_result.get("message"):
                        result["message"] = step_result["message"]
                    for key in ("target_object", "final_error_m"):
                        result.pop(key, None)
                        if key in step_result:
                            result[key] = step_result[key]
                    if step_result["status"] != "succeeded":
                        break
                session.emit({"type": "task_progress", "steps": task_steps,
                              "index": result["completed_steps"], "completed": result["completed_steps"],
                              "engine": "langgraph"})
        except Exception as exc:
            result.update(status="error", message=str(exc))
        result["final_position"] = np.asarray(session.executor.position).tolist()
        message = result.get("message")
        if not message:
            message = (f"任务完成，共完成 {result['completed_steps']} 个步骤。" if result["status"] == "succeeded"
                       else f"任务{result['status']}，已完成 {result['completed_steps']} 个步骤。")
        session.remember("assistant", {"status": result["status"], "message": message,
                                       "completed_steps": result["completed_steps"], "position": result["final_position"]})
        session.emit({"type": "result", "result": result})
        return result
