import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from main import local_math_answer


CALLS = []


class MockLocalModelHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length).decode("utf-8"))
        CALLS.append(request)
        prompt = request["messages"][-1]["content"]
        answer = f"<think>private reasoning</think>mock local answer for: {prompt[:60]}"
        body = {
            "id": "local-test",
            "object": "chat.completion",
            "model": "qwen3.5-2b-local",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        }
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_mock_server(port):
    ThreadingHTTPServer(("127.0.0.1", port), MockLocalModelHandler).serve_forever()


def main():
    if local_math_answer("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?") != "144":
        raise SystemExit("deterministic inventory solver failed")
    if local_math_answer("Calculate 18% of 250 and add 17.") != "62":
        raise SystemExit("deterministic percentage solver failed")

    tasks = [
        {"task_id": 101, "prompt": "What does a CPU cache do?"},
        {"task_id": "math", "prompt": "Calculate 18% of 250 and add 17."},
        {"task_id": "sentiment", "prompt": "Classify the sentiment and justify it: The service was fast but the meal was cold."},
        {"task_id": "summary", "prompt": "Summarize in one sentence: Open standards help teams integrate software reliably."},
        {"task_id": "ner", "prompt": "Extract named entities: Dr. Maya Chen met AMD engineers in Austin on July 2, 2026."},
        {"task_id": "debug", "prompt": "Fix the bug in this Python code: def add(a,b): return a-b"},
        {"task_id": "logic", "prompt": "Solve this logic puzzle: Ana is older than Bo. Bo is older than Cy. Who is youngest?"},
        {"task_id": "codegen", "prompt": "Write a Python function is_even(n) that returns True for even integers."},
    ]

    port = 8011
    threading.Thread(target=run_mock_server, args=(port,), daemon=True).start()
    time.sleep(0.3)

    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        input_path = root / "tasks.json"
        output_path = root / "results.json"
        input_path.write_text(json.dumps(tasks), encoding="utf-8")

        environment = os.environ.copy()
        environment.pop("FIREWORKS_API_KEY", None)
        environment.pop("FIREWORKS_BASE_URL", None)
        environment.pop("ALLOWED_MODELS", None)
        environment["INPUT_PATH"] = str(input_path)
        environment["OUTPUT_PATH"] = str(output_path)
        environment["LOCAL_SERVER_URL"] = f"http://127.0.0.1:{port}"

        result = subprocess.run([sys.executable, "main.py"], env=environment, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            raise SystemExit(f"main.py failed with exit code {result.returncode}")

        output = json.loads(output_path.read_text(encoding="utf-8"))
        if len(output) != len(tasks):
            raise SystemExit(f"expected {len(tasks)} results, got {len(output)}")
        if [item["task_id"] for item in output] != [task["task_id"] for task in tasks]:
            raise SystemExit("task IDs or ordering changed")
        if any(set(item) != {"task_id", "answer"} or not item["answer"] for item in output):
            raise SystemExit("each result must contain only non-empty task_id and answer fields")
        if any("<think>" in item["answer"] for item in output):
            raise SystemExit("private reasoning tags leaked into output")
        if len(CALLS) != len(tasks) - 1:
            raise SystemExit(f"expected one task to use deterministic math and all others to call Qwen, got {len(CALLS)} calls")
        if any(call.get("model") != "qwen3.5-2b-local" for call in CALLS):
            raise SystemExit("unexpected local model name")

    print("SUCCESS: local-only Track 1 contract test passed with zero Fireworks calls.")


if __name__ == "__main__":
    main()
