import threading
import unittest
import numpy as np

from session import NavigationSession
from planning import smoke_plan
from test_session import FakeWorld, FakeExecutor, scene
from task_graph import TaskWorkflow, validate_tasks


class TaskGraphTests(unittest.TestCase):
    def test_only_task_summary_is_published(self):
        for steps in ([{"action": "navigate", "target": "chair"}],
                      [{"action": "navigate", "target": "chair"},
                       {"action": "navigate", "target": "table"}]):
            events = []
            session = NavigationSession(FakeWorld(), FakeExecutor(), scene(), emit=events.append)
            result = TaskWorkflow(session).run_turn("导航测试", steps=steps)
            self.assertEqual(result["status"], "succeeded")
            replies = [e for e in events if e["type"] == "message" and e["role"] == "assistant"]
            self.assertEqual(len(replies), 1)
            self.assertEqual(replies[0]["content"]["completed_steps"], len(steps))
            # Internal outcomes remain available to follow-up model calls.
            self.assertTrue(any(m["content"].get("target_object") == "chair"
                                for m in session.history if m["role"] == "assistant"))

    def test_single_target_skips_decomposition(self):
        class Planner:
            starts = []
            def plan_tasks(self, instruction, history=None):
                raise AssertionError("Natural language must not call task decomposition")
            def plan(self, instruction, current_scene, start, feedback, history):
                self.starts.append(current_scene["nodes"][start])
                target = "saved_task_start" if "saved_task_start" in instruction else "chair"
                return smoke_plan(current_scene, start, target)
        planner = Planner()
        session = NavigationSession(FakeWorld(), FakeExecutor(), scene(), planner)
        result = TaskWorkflow(session).run_turn("去椅子旁边")
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(result["mode"], "agent")
        self.assertEqual(result["completed_steps"], 1)
        np.testing.assert_allclose(planner.starts, [[0., 0., 0.]])

    def test_natural_language_multi_rejected_without_moving(self):
        session = NavigationSession(FakeWorld(), FakeExecutor(), scene(), object())
        result = TaskWorkflow(session).run_turn("先去椅子，再去桌子，最后回来")
        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["completed_steps"], 0)
        self.assertEqual(result["start_position"], result["final_position"])

    def test_multi_target_return_and_thread_ownership(self):
        owner = threading.get_ident()
        class Executor(FakeExecutor):
            def execute_segment(self, *args, **kwargs):
                if threading.get_ident() != owner:
                    raise AssertionError("Simulation moved to graph thread")
                return super().execute_segment(*args, **kwargs)
        session = NavigationSession(FakeWorld(), Executor(), scene())
        workflow = TaskWorkflow(session)
        plan = [{"action": "navigate", "target": "chair"}, {"action": "navigate", "target": "table"},
                {"action": "return", "location": "task_start"}]
        result = workflow.run_turn("multi", steps=plan)
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(result["completed_steps"], 3)
        self.assertEqual(result["engine"], "langgraph")
        np.testing.assert_allclose(result["start_position"], result["final_position"])

    def test_task_start_is_captured_for_each_new_task(self):
        session = NavigationSession(FakeWorld(), FakeExecutor(), scene())
        workflow = TaskWorkflow(session)
        workflow.run_turn("first", target="chair")
        result = workflow.run_turn("out and back", steps=[{"action": "navigate", "target": "table"},
                                                         {"action": "return", "location": "task_start"}])
        self.assertEqual(result["status"], "succeeded", result)
        self.assertEqual(result["start_position"], [2.,0.,0.])
        np.testing.assert_allclose(result["final_position"], [2.,0.,0.])

    def test_failure_stops_remaining_steps(self):
        class Blocked(FakeExecutor):
            def execute_segment(self, *args, **kwargs):
                return {"status": "blocked"}
        session = NavigationSession(FakeWorld(), Blocked(), scene())
        result = TaskWorkflow(session).run_turn("multi", steps=[{"action": "navigate", "target": "chair"},
                                                               {"action": "navigate", "target": "table"}])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["completed_steps"], 0)

    def test_unsupported_action_rejected(self):
        with self.assertRaises(ValueError):
            validate_tasks({"steps": [{"action": "circle", "laps": 3}]})

    def test_cancel_before_action(self):
        session = NavigationSession(FakeWorld(), FakeExecutor(), scene())
        result = TaskWorkflow(session).run_turn("cancel", target="chair", should_stop=lambda: True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["completed_steps"], 0)


if __name__ == "__main__":
    unittest.main()
