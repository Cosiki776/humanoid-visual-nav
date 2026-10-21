"""Boundary tests: malformed model plans must never reach the simulator."""
import unittest
import io
import json
from unittest.mock import patch

from planning import ModelPlanner, smoke_plan, validate_plan


class PlanValidationTests(unittest.TestCase):
    def setUp(self):
        self.scene = {"nodes": {"a": [0,0,0], "b": [1,0,0], "c": [1,0,1], "d": [2,0,1]},
                      "edges": [["a", "b"], ["b", "c"], ["c", "d"]],
                      "objects": [{"id": "chair", "goal_nodes": ["c"]}]}

    def test_valid_detour(self):
        plan = {"target_object": "chair", "route": ["a", "b", "c"]}
        self.assertEqual(validate_plan(plan, self.scene, "a"), plan)

    def test_rejects_teleport_wrong_target_and_cycles(self):
        for route in [["a", "c"], ["a", "b"], ["b", "c"], ["a", "b", "a", "b", "c"],
                      ["a", "missing", "c"], ["a", {}, "c"], []]:
            with self.subTest(route=route), self.assertRaises(ValueError):
                validate_plan({"target_object": "chair", "route": route}, self.scene, "a")

    def test_rejects_malformed_target(self):
        with self.assertRaises(ValueError):
            validate_plan({"target_object": [], "route": ["a"]}, self.scene, "a")

    def test_baseline_is_explicit_and_reachable(self):
        plan = smoke_plan(self.scene, "a", "chair")
        self.assertEqual(plan["planner"], "offline_bfs_smoke_test")
        validate_plan(plan, self.scene, "a")
        self.scene["edges"] = [["a", "b"]]
        with self.assertRaises(ValueError):
            smoke_plan(self.scene, "a", "chair")

    def test_model_response_and_feedback_contract(self):
        plan = {"target_object": "chair", "route": ["a", "b", "c"]}
        body = {"choices": [{"delta": {"content": json.dumps(plan)}, "finish_reason": "stop"}]}
        with patch.dict("os.environ", {"NAV_LLM_BASE_URL": "http://localhost:8000/v1",
                                       "NAV_LLM_MODEL": "test-model", "NAV_LLM_API_KEY": ""}), \
                patch("planning.request.urlopen", return_value=io.BytesIO(('data: '+json.dumps(body)+'\n\ndata: [DONE]\n\n').encode())) as http:
            response = ModelPlanner().plan("走到椅子旁边", self.scene, "a", {"status": "invalid_plan"})
            validate_plan(response, self.scene, "a")
            req = http.call_args.args[0]
            self.assertTrue(json.loads(req.data)["stream"])
            self.assertEqual(req.full_url, "http://localhost:8000/v1/chat/completions")
            context = json.loads(json.loads(req.data)["messages"][1]["content"])
            self.assertEqual(context["feedback"]["status"], "invalid_plan")
            self.assertEqual(context["current_node"], "a")

    def test_missing_model_configuration_fails_explicitly(self):
        with patch.dict("os.environ", {"NAV_LLM_BASE_URL": "", "NAV_LLM_MODEL": ""}), \
                self.assertRaises(ValueError):
            ModelPlanner()


if __name__ == "__main__":
    unittest.main()
