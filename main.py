import ast
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")
MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/models/Qwen3.5-2B-Q4_K_M.gguf")
LLAMA_SERVER_PATH = os.getenv("LLAMA_SERVER_PATH", "/opt/llama/llama-server")
LOCAL_SERVER_URL = os.getenv("LOCAL_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "qwen3.5-2b-local")
SERVER_STARTUP_SECONDS = min(max(float(os.getenv("SERVER_STARTUP_SECONDS", "120")), 20.0), 180.0)
TASK_TIMEOUT_SECONDS = min(max(float(os.getenv("TASK_TIMEOUT_SECONDS", "75")), 20.0), 120.0)
TOTAL_RUNTIME_SECONDS = min(max(float(os.getenv("TOTAL_RUNTIME_SECONDS", "540")), 120.0), 570.0)

if INPUT_PATH == "/input/tasks.json" and not os.path.exists(INPUT_PATH) and os.path.exists("input/tasks.json"):
    INPUT_PATH = "input/tasks.json"
    OUTPUT_PATH = "output/results.json"


@dataclass(frozen=True)
class TaskProfile:
    category: str
    guidance: str
    max_tokens: int


GLOBAL_SYSTEM_PROMPT = (
    "You are an accurate general-purpose assistant. Solve the user's task completely. "
    "Follow every format, length, language, and explanation requirement in the original prompt. "
    "Answer every part of multi-part questions. Be concise, but never remove information the user requests. "
    "Do not reveal chain-of-thought or mention these instructions. Return only the final response."
)

CATEGORY_GUIDANCE = {
    "factual": (
        "Give an accurate factual answer. Explain definitions or mechanisms when asked, and answer all parts."
    ),
    "math": (
        "Calculate carefully and verify the result. Include units, rounding, or working only when requested."
    ),
    "sentiment": (
        "Use the requested sentiment labels exactly. Mixed sentiment is valid when supported. "
        "Include a justification when the prompt asks for one."
    ),
    "summary": (
        "Summarize only the supplied text and obey sentence, word, bullet, and style constraints exactly."
    ),
    "ner": (
        "Extract named entities faithfully, preserve their surface forms, and label their types in the requested format."
    ),
    "debug": (
        "Identify the actual defect and provide a complete corrected implementation. "
        "Include an explanation only when requested or needed by the task."
    ),
    "logic": (
        "Apply every stated constraint, check the conclusion, and return the requested conclusion and format."
    ),
    "codegen": (
        "Write complete, correct, executable code that satisfies every stated requirement and edge case."
    ),
}

MAX_TOKENS = {
    "factual": 180,
    "math": 180,
    "sentiment": 120,
    "summary": 240,
    "ner": 240,
    "debug": 420,
    "logic": 220,
    "codegen": 420,
}

CATEGORY_PATTERNS = {
    "sentiment": (
        r"\bsentiment\b",
        r"\bpositive\s*[,/]\s*negative\b",
        r"\bclassify (?:the )?(?:tone|emotion|review)\b",
        r"\bemotional tone\b",
    ),
    "summary": (
        r"\bsummari[sz]e\b",
        r"\bsummary\b",
        r"\bcondense\b",
        r"\btl;?dr\b",
        r"\bboil .{0,30} down\b",
        r"\bmain (?:points?|takeaways?|idea)\b",
    ),
    "ner": (
        r"\bnamed entit",
        r"\bextract .{0,30}(?:entities|people|persons?|organi[sz]ations?|locations?|dates?)\b",
        r"\b(?:identify|list|tag) .{0,20}(?:entities|people|organi[sz]ations?|locations?|dates?)\b",
    ),
    "debug": (
        r"\bdebug\b",
        r"\b(?:fix|find|identify) (?:the|this|a) (?:bug|error)\b",
        r"\bfix (?:this|the) (?:code|function|program|snippet)\b",
        r"\b(?:syntax error|traceback|exception|failing test|infinite loop)\b",
        r"\bwhat(?:'s| is) wrong with .{0,20}(?:code|function|program)\b",
    ),
    "codegen": (
        r"\b(?:write|implement|create|generate|build|define) .{0,40}(?:function|class|method|program|script|algorithm|code)\b",
        r"\b(?:function|code|program|script|method) (?:that|to)\b",
    ),
    "logic": (
        r"\b(?:logic puzzle|deductive|constraint-based|riddle|truth.teller|liar)\b",
        r"\b(?:who owns|who is|must be true|can be inferred)\b",
        r"\b(?:older than|younger than|left of|right of)\b",
        r"\beach .{0,30}(?:different|exactly one)\b",
    ),
    "math": (
        r"\b(?:calculate|compute|solve|evaluate|percentage|percent|ratio|probability|equation|average)\b",
        r"\b(?:how many|how much|remain|remaining|total cost|discount|interest)\b",
        r"\d+\s*%",
        r"\d+\s*(?:[+*/-]|plus|minus|times|divided by)\s*\d+",
    ),
}

CATEGORY_PRIORITY = ("debug", "codegen", "sentiment", "ner", "summary", "logic", "math")
COMPILED_PATTERNS = {
    category: tuple(re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns)
    for category, patterns in CATEGORY_PATTERNS.items()
}


def classify_task(prompt: str) -> str:
    for category in CATEGORY_PRIORITY:
        if any(pattern.search(prompt) for pattern in COMPILED_PATTERNS[category]):
            return category
    return "factual"


def build_profile(prompt: str) -> TaskProfile:
    category = classify_task(prompt)
    return TaskProfile(category, CATEGORY_GUIDANCE[category], MAX_TOKENS[category])


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.8f}".rstrip("0").rstrip(".")


def safe_arithmetic(expression: str) -> float | None:
    expression = expression.replace("^", "**").replace("x", "*").replace("X", "*")
    if not re.fullmatch(r"[\d\s+*/().%-]+", expression):
        return None

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )
    if any(not isinstance(node, allowed_nodes) for node in ast.walk(tree)):
        return None
    try:
        return float(eval(compile(tree, "<arithmetic>", "eval"), {"__builtins__": {}}, {}))
    except (ArithmeticError, TypeError, ValueError):
        return None


def local_math_answer(prompt: str) -> str | None:
    text = " ".join(prompt.lower().split())

    inventory = re.search(
        r"(?:has|starts? with)\s+(\d+(?:\.\d+)?)\s+(?:items?|units?|products?).*?"
        r"(?:sells?|sold)\s+(\d+(?:\.\d+)?)\s*%.*?"
        r"(?:and|then)\s+(\d+(?:\.\d+)?)\s+(more\s+than\s+(?:monday|the first day)|more\s+on\s+\w+|"
        r"(?:items?\s+)?on\s+\w+).*?"
        r"(?:remain|remaining|left)",
        text,
    )
    if inventory:
        starting, percentage, later = map(float, inventory.groups()[:3])
        first_day = starting * percentage / 100
        phrase = inventory.group(4)
        second_day = first_day + later if phrase.startswith("more than") else later
        return format_number(starting - first_day - second_day)

    percentage = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)", text)
    if percentage:
        percent, base = map(float, percentage.groups())
        value = base * percent / 100
        suffix = text[percentage.end():]
        addition = re.search(r"\b(?:add|plus)\s+(-?\d+(?:\.\d+)?)", suffix)
        subtraction = re.search(r"\b(?:subtract|minus)\s+(-?\d+(?:\.\d+)?)", suffix)
        if addition:
            value += float(addition.group(1))
        elif subtraction:
            value -= float(subtraction.group(1))
        return format_number(value)

    direct = re.search(
        r"(?:calculate|compute|evaluate|what is|find)\s+([\d\s+*/().%xX^-]+?)(?:\?|$)",
        text,
    )
    if direct:
        value = safe_arithmetic(direct.group(1).strip())
        if value is not None:
            return format_number(value)
    return None


def read_tasks() -> list[dict[str, Any]]:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8-sig") as file:
        tasks = json.load(file)

    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Input JSON must be a non-empty list of task objects.")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {index} is not an object.")
        if "task_id" not in task:
            raise ValueError(f"Task at index {index} must contain task_id.")
        if not any(key in task for key in ("prompt", "question", "input", "text")):
            raise ValueError(f"Task at index {index} must contain prompt text.")
    return tasks


def task_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        value = task.get(key)
        if value is not None:
            return str(value)
    raise ValueError("Task is missing prompt text.")


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def server_is_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{LOCAL_SERVER_URL}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_local_server() -> tuple[subprocess.Popen[Any] | None, Any | None]:
    if os.getenv("LOCAL_SERVER_URL"):
        logger.info("Using externally supplied local model server at %s", LOCAL_SERVER_URL)
        return None, None

    if not os.path.exists(LLAMA_SERVER_PATH):
        raise FileNotFoundError(f"llama-server not found: {LLAMA_SERVER_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Local model not found: {MODEL_PATH}")

    server_log = open("/tmp/llama-server.log", "w", encoding="utf-8")
    command = [
        LLAMA_SERVER_PATH,
        "--model", MODEL_PATH,
        "--alias", LOCAL_MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", "8080",
        "--ctx-size", "4096",
        "--parallel", "1",
        "--threads", "2",
        "--threads-batch", "2",
        "--batch-size", "256",
        "--ubatch-size", "128",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--reasoning", "off",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "--no-webui",
        "--offline",
    ]
    logger.info("Starting bundled Qwen3.5-2B model on 2 CPU threads")
    process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)

    deadline = time.monotonic() + SERVER_STARTUP_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            server_log.flush()
            raise RuntimeError(f"Local model server exited with code {process.returncode}; see /tmp/llama-server.log")
        if server_is_ready():
            logger.info("Local model server is ready")
            return process, server_log
        time.sleep(1)

    process.terminate()
    raise TimeoutError("Local model server did not become ready in time")


def stop_local_server(process: subprocess.Popen[Any] | None, server_log: Any | None) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    if server_log is not None:
        server_log.close()


def clean_answer(answer: str) -> str:
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
    answer = re.sub(r"^</?think>\s*", "", answer, flags=re.IGNORECASE).strip()
    if answer.startswith("```") and answer.endswith("```"):
        answer = re.sub(r"^```[A-Za-z0-9_+.-]*\s*", "", answer)
        answer = re.sub(r"\s*```$", "", answer).strip()
    return answer


def run_task(prompt: str, profile: TaskProfile, timeout: float) -> str:
    system_prompt = f"{GLOBAL_SYSTEM_PROMPT}\n\nTask guidance: {profile.guidance}"
    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": profile.max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "seed": 7,
    }
    response = post_json(f"{LOCAL_SERVER_URL}/v1/chat/completions", payload, timeout)
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Local model returned no choices")
    answer = clean_answer(str(choices[0].get("message", {}).get("content", "")))
    if not answer:
        raise RuntimeError("Local model returned an empty answer")
    return answer


def write_results(results: list[dict[str, Any]]) -> None:
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    clean_results = [{"task_id": item["task_id"], "answer": item["answer"]} for item in results]
    temporary_path = f"{OUTPUT_PATH}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(clean_results, file, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary_path, OUTPUT_PATH)


def run() -> int:
    started_at = time.monotonic()
    process: subprocess.Popen[Any] | None = None
    server_log: Any | None = None

    try:
        tasks = read_tasks()
        process, server_log = start_local_server()
        results: list[dict[str, Any]] = []

        for task in tasks:
            elapsed = time.monotonic() - started_at
            remaining = TOTAL_RUNTIME_SECONDS - elapsed
            if remaining < 10:
                raise TimeoutError("Not enough time remains to complete all tasks")

            prompt = task_prompt(task)
            profile = build_profile(prompt)
            answer = local_math_answer(prompt) if profile.category == "math" else None
            if answer is not None:
                logger.info("Processing task %s with deterministic local math", task["task_id"])
            else:
                logger.info("Processing task %s with Qwen as %s", task["task_id"], profile.category)
                answer = run_task(prompt, profile, min(TASK_TIMEOUT_SECONDS, remaining - 5))
            results.append({"task_id": task["task_id"], "answer": answer})

        write_results(results)
        logger.info(
            "Wrote %d local-model results to %s in %.1f seconds; Fireworks calls: 0",
            len(results),
            OUTPUT_PATH,
            time.monotonic() - started_at,
        )
        return 0
    except Exception as exc:
        logger.exception("Local agent failed: %s", exc)
        return 1
    finally:
        stop_local_server(process, server_log)


if __name__ == "__main__":
    sys.exit(run())
