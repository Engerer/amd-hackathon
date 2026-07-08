import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory


CALLS = []
ALLOWED_MODELS = [
    "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
]


class MockFireworksHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length).decode("utf-8"))
        model = request.get("model")
        messages = request.get("messages", [])
        user_prompt = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")

        CALLS.append(request)

        if model not in ALLOWED_MODELS:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "model not allowed"}}).encode("utf-8"))
            return

        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": f"mock answer for: {user_prompt[:80]}",
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))


def run_mock_server(port):
    server = ThreadingHTTPServer(("127.0.0.1", port), MockFireworksHandler)
    server.serve_forever()


def main():
    tasks = [
        {"task_id": 101, "prompt": "What does a CPU cache do?"},
        {"task_id": "math", "prompt": "Calculate 18% of 250 and add 17."},
        {"task_id": "sentiment", "prompt": "Classify the sentiment: The service was fast but the meal was cold."},
        {"task_id": "summary", "prompt": "Summarize in one sentence: Open standards help teams integrate software reliably."},
        {"task_id": "ner", "prompt": "Extract named entities: Dr. Maya Chen met AMD engineers in Austin on July 2, 2026."},
        {"task_id": "debug", "prompt": "Fix the bug in this Python code: def add(a,b): return a-b"},
        {"task_id": "logic", "prompt": "Solve this logic puzzle: Ana is older than Bo. Bo is older than Cy. Who is youngest?"},
        {"task_id": "codegen", "prompt": "Write a Python function is_even(n) that returns True for even integers."},
    ]

    port = 8000
    thread = threading.Thread(target=run_mock_server, args=(port,), daemon=True)
    thread.start()
    time.sleep(0.5)

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_path = input_dir / "tasks.json"
        results_path = output_dir / "results.json"
        input_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env["FIREWORKS_API_KEY"] = "mock-key"
        env["FIREWORKS_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["ALLOWED_MODELS"] = ",".join(ALLOWED_MODELS)
        env["MAX_CONCURRENCY"] = "4"
        env["INPUT_PATH"] = str(input_path)
        env["OUTPUT_PATH"] = str(results_path)

        result = subprocess.run([sys.executable, "main.py"], env=env, capture_output=True, text=True)
        print("--- stdout ---")
        print(result.stdout)
        print("--- stderr ---")
        print(result.stderr)

        if result.returncode != 0:
            raise SystemExit(f"main.py failed with exit code {result.returncode}")
        if not results_path.exists():
            raise SystemExit("results.json was not created")

        output = json.loads(results_path.read_text(encoding="utf-8"))
        if len(output) != len(tasks):
            raise SystemExit(f"expected {len(tasks)} results, got {len(output)}")
        if {json.dumps(item["task_id"]) for item in output} != {json.dumps(task["task_id"]) for task in tasks}:
            raise SystemExit("result task_ids do not match input task_ids")
        expected_fields = {"task_id", "answer"}
        if any(set(item) != expected_fields or not item["answer"] for item in output):
            raise SystemExit("each result must contain non-empty task_id and answer fields only")
        if not (1 <= len(CALLS) <= len(tasks) * 4):
            raise SystemExit(f"unexpected Fireworks call count: {len(CALLS)}")
        if any(call["model"] not in ALLOWED_MODELS for call in CALLS):
            raise SystemExit("agent called a model outside ALLOWED_MODELS")
        numeric_result = next(item for item in output if item["task_id"] == 101)
        if not isinstance(numeric_result["task_id"], int):
            raise SystemExit("numeric task_id was not preserved")

    print("SUCCESS: Track 1 contract test passed.")


if __name__ == "__main__":
    main()
