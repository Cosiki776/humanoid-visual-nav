"""Persistent pose, conversational context, and failure isolation contracts."""
import unittest
import numpy as np
from planning import smoke_plan
from session import NavigationSession


class FakeWorld:
    def direct(self, a, b):
        return True


class FakeExecutor:
    def __init__(self):
        self.position = np.array([0., 0., 0.])

    def execute_segment(self, point, should_stop):
        if should_stop():
            return {"status": "cancelled"}
        self.position = np.array(point, dtype=float)
        return {"status": "succeeded", "position": self.position.tolist()}


def scene():
    return {"nodes": {"a": [0,0,0], "b": [1,0,0], "c": [2,0,0], "d": [3,0,0]},
            "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
            "objects": [{"id": "chair", "goal_nodes": ["c"]}, {"id": "table", "goal_nodes": ["a"]}]}


class SessionTests(unittest.TestCase):
    def test_second_turn_begins_at_previous_actual_pose(self):
        executor = FakeExecutor()
        session = NavigationSession(FakeWorld(), executor, scene())
        first = session.run_turn("椅子", "chair")
        second = session.run_turn("桌子", "table")
        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(second["status"], "succeeded")
        self.assertEqual(second["start_position"], first["final_position"])
        self.assertNotEqual(first["start_position"], second["start_position"])
        self.assertEqual(len(session.history), 4)
        self.assertNotIn("session_current", session.scene["nodes"])

    def test_clarification_and_completion_are_passed_to_next_model_call(self):
        class Planner:
            calls = []
            def plan(self, instruction, current_scene, start, feedback, history):
                self.calls.append(history)
                if len(self.calls) == 1:
                    return {"status": "needs_clarification", "message": "哪一把椅子？"}
                return smoke_plan(current_scene, start, "chair")
        planner = Planner()
        session = NavigationSession(FakeWorld(), FakeExecutor(), scene(), planner)
        self.assertEqual(session.run_turn("去椅子旁边")["status"], "needs_clarification")
        self.assertEqual(session.run_turn("最近的那把")["status"], "succeeded")
        self.assertEqual(planner.calls[1][-1]["content"]["message"], "哪一把椅子？")
        session.run_turn("再去那里")
        self.assertEqual(planner.calls[2][-1]["content"]["target_object"], "chair")

    def test_model_failure_does_not_destroy_session(self):
        class Planner:
            def plan(self, *args, **kwargs):
                raise RuntimeError("endpoint unavailable")
        session = NavigationSession(FakeWorld(), FakeExecutor(), scene(), Planner())
        self.assertEqual(session.run_turn("go")["status"], "error")
        self.assertEqual(session.run_turn("offline", "chair")["status"], "succeeded")

    def test_cancellation_preserves_pose(self):
        session = NavigationSession(FakeWorld(), FakeExecutor(), scene())
        result = session.run_turn("stop", "chair", lambda: True)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["start_position"], result["final_position"])


if __name__ == "__main__":
    unittest.main()
