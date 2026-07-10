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


LOCAL_CALLS = []
FIREWORKS_CALLS = []
ALLOWED_MODELS = ["minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it"]


def prompt_answer(prompt):
    lowered = prompt.lower()
    if "capital of australia" in lowered:
        return "Canberra is near Lake Macquarie."
    if "sentiment" in lowered:
        return "Mixed. It praises the battery but criticizes the screen."
    if "summarize" in lowered:
        return "Open standards reduce integration work and vendor lock-in through shared interfaces."
    if "named entities" in lowered:
        return "Maria; Fireworks AI; Berlin"
    if "get_max" in lowered:
        return "def get_max(nums):\n    return max(nums)"
    if "who owns the cat" in lowered:
        return "Therefore, Sam owns the cat."
    if "second-largest" in lowered:
        return "def second_largest(values):\n    unique = sorted(set(values), reverse=True)\n    if len(unique) < 2: raise ValueError\n    return unique[1]"
    return "Local answer"


class MockLocalHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        if self.path != "/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        LOCAL_CALLS.append(request)
        prompt = request["messages"][-1]["content"]
        answer = prompt_answer(prompt)
        token_logprobs = [
            {"token": token, "logprob": -0.12, "top_logprobs": []}
            for token in answer.split()
        ]
        self.send_json({
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": answer},
                "logprobs": {"content": token_logprobs},
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": len(token_logprobs), "total_tokens": 20 + len(token_logprobs)},
        })

    def send_json(self, body):
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class MockFireworksHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        FIREWORKS_CALLS.append(request)
        prompt = request["messages"][-1]["content"].lower()
        if "capital of australia" in prompt:
            answer = "Canberra, near Lake Burley Griffin."
        elif "named entities" in prompt:
            answer = "Person: Maria Sanchez\nOrganization: Fireworks AI\nLocation: Berlin\nDate: March"
        else:
            answer = "Escalated answer"
        encoded = json.dumps({
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve(port, handler):
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


def main():
    if local_math_answer("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?") != "144":
        raise SystemExit("deterministic inventory solver failed")
    if local_math_answer("Calculate 18% of 250 and add 17.") != "62":
        raise SystemExit("deterministic percentage solver failed")

    tasks = [
        {"task_id": "factual", "prompt": "What is the capital of Australia, and what body of water is it near?"},
        {"task_id": "math", "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?"},
        {"task_id": "sentiment", "prompt": "Classify the sentiment and justify it: The battery is great, but the screen scratches easily."},
        {"task_id": "summary", "prompt": "Summarize in one sentence: Open standards reduce integration work and vendor lock-in through shared interfaces."},
        {"task_id": "ner", "prompt": "Extract all named entities: Maria Sanchez joined Fireworks AI in Berlin last March."},
        {"task_id": "debug", "prompt": "Fix the bug in this Python code: def get_max(nums): return nums[0]"},
        {"task_id": "logic", "prompt": "Jo owns the dog. Sam does not own the bird. Who owns the cat?"},
        {"task_id": "codegen", "prompt": "Write a Python function returning the second-largest distinct value."},
    ]

    threading.Thread(target=serve, args=(8011, MockLocalHandler), daemon=True).start()
    threading.Thread(target=serve, args=(8012, MockFireworksHandler), daemon=True).start()
    time.sleep(0.3)

    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        input_path = root / "tasks.json"
        output_path = root / "results.json"
        input_path.write_text(json.dumps(tasks), encoding="utf-8")
        environment = os.environ.copy()
        environment["INPUT_PATH"] = str(input_path)
        environment["OUTPUT_PATH"] = str(output_path)
        environment["LOCAL_SERVER_URL"] = "http://127.0.0.1:8011"
        environment["FIREWORKS_API_KEY"] = "mock-harness-key"
        environment["FIREWORKS_BASE_URL"] = "http://127.0.0.1:8012/v1"
        environment["ALLOWED_MODELS"] = ",".join(ALLOWED_MODELS)

        result = subprocess.run([sys.executable, "main.py"], env=environment, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            raise SystemExit(f"main.py failed with exit code {result.returncode}")

        output = json.loads(output_path.read_text(encoding="utf-8"))
        if len(output) != len(tasks) or any(set(item) != {"task_id", "answer"} for item in output):
            raise SystemExit("invalid Track 1 result schema")
        by_id = {item["task_id"]: item["answer"] for item in output}
        if by_id["math"] != "144":
            raise SystemExit("deterministic math result was not used")
        if "Lake Burley Griffin" not in by_id["factual"]:
            raise SystemExit("factual task did not use Fireworks correction")
        if "Maria Sanchez" not in by_id["ner"]:
            raise SystemExit("NER task did not use Fireworks correction")
        if len(FIREWORKS_CALLS) != 2:
            raise SystemExit(f"expected two Fireworks escalations, got {len(FIREWORKS_CALLS)}")
        if any(call["model"] not in ALLOWED_MODELS for call in FIREWORKS_CALLS):
            raise SystemExit("called a model outside ALLOWED_MODELS")
        if len(LOCAL_CALLS) != 9:
            raise SystemExit(f"expected nine local samples, got {len(LOCAL_CALLS)}")

    print("SUCCESS: hybrid Track 1 contract and routing test passed.")


if __name__ == "__main__":
    main()
