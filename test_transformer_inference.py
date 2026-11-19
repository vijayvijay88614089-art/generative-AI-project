import unittest
from io import BytesIO
from unittest.mock import patch

import torch
from modules.transformer_inference import _execution_device, generate_local, model_capacity_status, ollama_model_status


class TransformerDeviceTests(unittest.TestCase):
    def test_skips_meta_and_disk_offload_entries(self):
        model = type("Model", (), {"hf_device_map": {"embed": "disk", "layer.0": "cpu"}})()
        self.assertEqual(_execution_device(model, torch), torch.device("cpu"))

    def test_uses_cpu_when_no_device_map_or_cuda(self):
        model = type("Model", (), {"hf_device_map": {}})()
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(_execution_device(model, torch), torch.device("cpu"))

    def test_converts_integer_device_map_entry_to_cuda_device(self):
        model = type("Model", (), {"hf_device_map": {"embed": 0}})()
        self.assertEqual(_execution_device(model, torch), torch.device("cuda:0"))

    def test_qwen_capacity_warning_for_low_memory_cpu(self):
        memory = type("Memory", (), {"available": 8 * 1024**3, "total": 8 * 1024**3})()
        with patch("torch.cuda.is_available", return_value=False), patch("psutil.virtual_memory", return_value=memory):
            ready, message = model_capacity_status("Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertFalse(ready)
        self.assertIn("quantized Qwen", message)

    def test_ollama_model_status_recognizes_pulled_model(self):
        payload = b'{"models":[{"name":"qwen2.5-coder:1.5b"}]}'
        with patch("modules.transformer_inference.urlopen", return_value=BytesIO(payload)):
            ready, message = ollama_model_status("qwen2.5-coder:1.5b")
        self.assertTrue(ready)
        self.assertEqual(message, "")

    def test_ollama_generation_uses_local_chat_api(self):
        payload = b'{"message":{"content":"local result"}}'
        with patch("modules.transformer_inference.urlopen", return_value=BytesIO(payload)) as request:
            output = generate_local("qwen2.5-coder:1.5b", "hello", max_new_tokens=10)
        self.assertEqual(output, "local result")
        self.assertEqual(request.call_args.args[0].full_url, "http://127.0.0.1:11434/api/chat")


if __name__ == "__main__":
    unittest.main()
