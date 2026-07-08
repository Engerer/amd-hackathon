import json
import os
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# Define a mock Fireworks API server
class MockFireworksHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging of HTTP requests to keep output clean
        return

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req_body = json.loads(post_data.decode('utf-8'))
            
            # Extract request details
            messages = req_body.get("messages", [])
            model = req_body.get("model", "")
            
            user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
            system_message = next((m["content"] for m in messages if m["role"] == "system"), "")
            
            # Generate a mock response
            response_text = f"[Mock Response using model {model}] Received system prompt: '{system_message}' and user prompt: '{user_message}'"
            
            res_body = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ]
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(res_body).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_mock_server(server_class=HTTPServer, handler_class=MockFireworksHandler, port=8000):
    server_address = ('127.0.0.1', port)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()

def main():
    # 1. Create mock inputs
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    mock_tasks = [
        {
            "task_id": "t_factual",
            "prompt": "What is the capital of France?"
        },
        {
            "task_id": "t_math",
            "prompt": "Calculate the sum of 15 and 27."
        },
        {
            "task_id": "t_sentiment",
            "prompt": "Classify the sentiment: I love this new product!"
        },
        {
            "task_id": "t_summary",
            "prompt": "Summarize this: The quick brown fox jumps over the lazy dog."
        },
        {
            "task_id": "t_ner",
            "prompt": "Extract person entities: John Doe went to Paris."
        },
        {
            "task_id": "t_debug",
            "prompt": "Fix the code snippet with a syntax error."
        },
        {
            "task_id": "t_logic",
            "prompt": "Solve this logic puzzle with constraints."
        },
        {
            "task_id": "t_codegen",
            "prompt": "Write a function to add two numbers."
        }
    ]
    
    with open("input/tasks.json", "w", encoding="utf-8") as f:
        json.dump(mock_tasks, f, indent=2)
        
    print("Mock tasks written to input/tasks.json")
    
    # 2. Start mock server in background thread
    server_port = 8000
    server_thread = threading.Thread(target=run_mock_server, args=(HTTPServer, MockFireworksHandler, server_port))
    server_thread.daemon = True
    server_thread.start()
    
    print(f"Mock Fireworks API server started at http://127.0.0.1:{server_port}")
    time.sleep(1)  # Allow server to start up
    
    # 3. Set environment variables
    env = os.environ.copy()
    env["FIREWORKS_API_KEY"] = "mock_api_key_123"
    env["FIREWORKS_BASE_URL"] = f"http://127.0.0.1:{server_port}/v1"
    env["ALLOWED_MODELS"] = "google/gemma-4-e2b-it,other-model"
    
    # 4. Run main.py as a subprocess
    print("Running main.py agent...")
    result = subprocess.run([sys.executable, "main.py"], env=env, capture_output=True, text=True)
    
    print("\n--- Agent stdout ---")
    print(result.stdout)
    print("--- Agent stderr ---")
    print(result.stderr)
    print("--------------------\n")
    
    if result.returncode != 0:
        print("FAIL: main.py exited with non-zero code.")
        sys.exit(1)
        
    # 5. Verify outputs
    if not os.path.exists("output/results.json"):
        print("FAIL: output/results.json was not created.")
        sys.exit(1)
        
    try:
        with open("output/results.json", "r", encoding="utf-8") as f:
            output_data = json.load(f)
    except Exception as e:
        print(f"FAIL: Failed to parse output/results.json: {e}")
        sys.exit(1)
        
    print("Output results:")
    print(json.dumps(output_data, indent=2))
    
    if len(output_data) != len(mock_tasks):
        print(f"FAIL: Number of tasks processed ({len(output_data)}) does not match input tasks ({len(mock_tasks)}).")
        sys.exit(1)
        
    # Check that all keys are correct
    for item in output_data:
        if "task_id" not in item or "answer" not in item:
            print(f"FAIL: Result entry is missing task_id or answer: {item}")
            sys.exit(1)
            
    print("\nSUCCESS: All verification checks passed!")

if __name__ == "__main__":
    main()
