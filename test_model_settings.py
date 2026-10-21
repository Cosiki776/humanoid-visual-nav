import json
import os
import unittest
from unittest.mock import patch

from model_settings import load_model_settings


class ModelSettingsTests(unittest.TestCase):
    def read(self, values, environment=None):
        with patch.dict(os.environ, environment or {}, clear=True), \
                patch("model_settings.CONFIG_PATH") as path:
            path.exists.return_value = True
            path.read_text.return_value = json.dumps(values)
            return load_model_settings()

    def test_file_config_and_empty_key(self):
        cfg = {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
               "model": "qwen3.8-max", "api_key": "", "timeout_seconds": 120}
        self.assertFalse(self.read(cfg)["configured"])
        cfg["api_key"] = "test-key-not-a-real-credential"
        loaded = self.read(cfg)
        self.assertTrue(loaded["configured"])
        self.assertEqual(loaded["model"], "qwen3.8-max")

    def test_environment_overrides_file(self):
        loaded = self.read({"base_url": "https://example.com/v1", "model": "file-model", "api_key": "file-key"},
                           {"NAV_LLM_BASE_URL": "http://localhost:8000/v1/", "NAV_LLM_MODEL": "local",
                            "NAV_LLM_API_KEY": ""})
        self.assertTrue(loaded["configured"])
        self.assertEqual(loaded["api_key"], "")
        self.assertEqual(loaded["model"], "local")
        self.assertEqual(loaded["base_url"], "http://localhost:8000/v1")

    def test_invalid_values(self):
        for cfg in [{"api_key": 123}, {"timeout_seconds": -1}, {"base_url": "file:///tmp/key"}, []]:
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                self.read(cfg)


if __name__ == "__main__":
    unittest.main()
