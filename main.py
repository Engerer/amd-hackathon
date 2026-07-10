import ast
import difflib
import json
import logging
import math
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
TASK_TIMEOUT_SECONDS = min(max(float(os.getenv("TASK_TIMEOUT_SECONDS", "70")), 20.0), 100.0)
TOTAL_RUNTIME_SECONDS = min(max(float(os.getenv("TOTAL_RUNTIME_SECONDS", "540")), 120.0), 570.0)

if INPUT_PATH == "/input/tasks.json" and not os.path.exists(INPUT_PATH) and os.path.exists("input/tasks.json"):
    INPUT_PATH = "input/tasks.json"
    OUTPUT_PATH = "output/results.json"


@dataclass(frozen=True)
class TaskProfile:
    category: str
    guidance: str
    max_tokens: int
    samples: int
    confidence_floor: float


@dataclass(frozen=True)
class LocalCandidate:
    answer: str
    confidence: float
    finish_reason: str


GLOBAL_SYSTEM_PROMPT = (
    "Solve the user's task accurately and completely. Follow every requested format, length, language, "
    "and explanation requirement. Answer all parts. Be concise and return only the final response."
)

CATEGORY_GUIDANCE = {
    "factual": "Verify facts and relationships. Explain when asked and answer every part.",
    "math": "Calculate carefully. Preserve requested units, precision, and working.",
    "sentiment": "Use the requested label set. Mixed is valid. Justify only when requested.",
    "summary": "Use only the supplied text and obey exact length, sentence, bullet, and style constraints.",
    "ner": "Preserve complete entity surface forms and label every requested type.",
    "debug": "Identify the real defect and provide a complete corrected implementation.",
    "logic": "Apply every constraint, verify the conclusion, and follow the requested format.",
    "codegen": "Return complete executable code satisfying the specification and edge cases.",
}

MAX_TOKENS = {
    "factual": 180,
    "math": 220,
    "sentiment": 120,
    "summary": 240,
    "ner": 240,
    "debug": 420,
    "logic": 220,
    "codegen": 420,
}

LOCAL_SAMPLES = {
    "factual": 1,
    "math": 1,
    "sentiment": 2,
    "summary": 1,
    "ner": 1,
    "debug": 1,
    "logic": 2,
    "codegen": 1,
}

CONFIDENCE_FLOORS = {
    "factual": 0.55,
    "math": 0.55,
    "sentiment": 0.38,
    "summary": 0.24,
    "ner": 0.40,
    "debug": 0.22,
    "logic": 0.32,
    "codegen": 0.22,
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

MODEL_EXCLUDE_HINTS = (
    "audio", "clip", "diffusion", "embed", "guard", "image", "moderation", "rerank", "tts", "vision", "whisper"
)

CATEGORY_MODEL_HINTS = {
    "factual": ("minimax", "gemma", "llama", "qwen", "kimi", "deepseek"),
    "math": ("minimax", "reason", "qwq", "qwen", "kimi", "deepseek", "llama"),
    "sentiment": ("gemma", "minimax", "llama", "qwen", "kimi"),
    "summary": ("gemma", "minimax", "llama", "qwen", "kimi"),
    "ner": ("minimax", "gemma", "llama", "qwen", "kimi"),
    "debug": ("kimi", "coder", "code", "qwen", "deepseek", "minimax"),
    "logic": ("minimax", "reason", "qwen", "kimi", "deepseek", "llama"),
    "codegen": ("kimi", "coder", "code", "qwen", "deepseek", "minimax"),
}


def classify_task(prompt: str) -> str:
    for category in CATEGORY_PRIORITY:
        if any(pattern.search(prompt) for pattern in COMPILED_PATTERNS[category]):
            return category
    return "factual"


def build_profile(prompt: str) -> TaskProfile:
    category = classify_task(prompt)
    return TaskProfile(
        category=category,
        guidance=CATEGORY_GUIDANCE[category],
        max_tokens=MAX_TOKENS[category],
        samples=LOCAL_SAMPLES[category],
        confidence_floor=CONFIDENCE_FLOORS[category],
    )


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
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant,
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
        r"(?:items?\s+)?on\s+\w+).*?(?:remain|remaining|left)",
        text,
    )
    if inventory:
        starting, percentage, later = map(float, inventory.groups()[:3])
        first_day = starting * percentage / 100
        second_day = first_day + later if inventory.group(4).startswith("more than") else later
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

    direct = re.search(r"(?:calculate|compute|evaluate|what is|find)\s+([\d\s+*/().%xX^-]+?)(?:\?|$)", text)
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
        if not isinstance(task, dict) or "task_id" not in task:
            raise ValueError(f"Task at index {index} must be an object containing task_id.")
        if not any(key in task for key in ("prompt", "question", "input", "text")):
            raise ValueError(f"Task at index {index} must contain prompt text.")
    return tasks


def task_prompt(task: dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "text"):
        if task.get(key) is not None:
            return str(task[key])
    raise ValueError("Task is missing prompt text.")


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def server_is_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{LOCAL_SERVER_URL}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def start_local_server() -> tuple[subprocess.Popen[Any] | None, Any | None]:
    if os.getenv("LOCAL_SERVER_URL"):
        logger.info("Using supplied local model server at %s", LOCAL_SERVER_URL)
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
    logger.info("Starting bundled Qwen3.5-2B on two CPU threads")
    process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + SERVER_STARTUP_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            server_log.flush()
            raise RuntimeError(f"Local model server exited with code {process.returncode}")
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


def token_confidence(response: dict[str, Any]) -> float:
    try:
        content = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return 0.0
    values = [float(item["logprob"]) for item in content if item.get("token") and item.get("logprob") is not None]
    if not values:
        return 0.0
    return max(0.0, min(1.0, math.exp(sum(values) / len(values))))


def run_local_sample(prompt: str, profile: TaskProfile, timeout: float, seed: int, temperature: float) -> LocalCandidate:
    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": f"{GLOBAL_SYSTEM_PROMPT}\n{profile.guidance}"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": profile.max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 20,
        "seed": seed,
        "logprobs": True,
        "top_logprobs": 3,
    }
    response = post_json(chat_endpoint(LOCAL_SERVER_URL), payload, timeout)
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Local model returned no choices")
    answer = clean_answer(str(choices[0].get("message", {}).get("content", "")))
    if not answer:
        raise RuntimeError("Local model returned an empty answer")
    return LocalCandidate(answer, token_confidence(response), str(choices[0].get("finish_reason", "")))


def candidate_key(answer: str, category: str) -> str:
    text = answer.strip().lower()
    if category == "sentiment":
        head = " ".join(text.split()[:12])
        if re.search(r"\bmixed\b", head):
            return "mixed"
        for label in ("positive", "negative", "neutral"):
            if re.search(rf"\b{label}\b", head):
                return label
    if category == "math":
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", text)
        if numbers:
            return numbers[-1].replace(" ", "")
    if category == "logic":
        therefore = re.findall(r"(?:therefore|thus|so),?\s+([^\n]+)", text)
        if therefore:
            text = therefore[-1]
        else:
            sentences = [part.strip() for part in re.split(r"[.!?\n]+", text) if part.strip()]
            if sentences:
                text = sentences[-1]
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def candidates_agree(candidates: list[LocalCandidate], category: str) -> bool:
    if len(candidates) < 2:
        return True
    keys = [candidate_key(candidate.answer, category) for candidate in candidates]
    if all(key == keys[0] for key in keys[1:]):
        return True
    return all(difflib.SequenceMatcher(None, keys[0], key).ratio() >= 0.82 for key in keys[1:])


def format_is_valid(prompt: str, answer: str, category: str) -> bool:
    if not answer.strip():
        return False
    if category == "sentiment" and candidate_key(answer, category) not in {"positive", "negative", "neutral", "mixed"}:
        return False
    if category in {"debug", "codegen"} and re.search(r"\b(?:function|class|method|python|javascript|code)\b", prompt, re.IGNORECASE):
        if not re.search(r"\b(?:def|class|function|return|const|let|var|public|private)\b", answer):
            return False
    if category == "summary" and re.search(r"\bexactly one sentence\b", prompt, re.IGNORECASE):
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", answer.strip()) if part]
        if len(sentences) != 1:
            return False
    return True


def escalation_reason(prompt: str, profile: TaskProfile, candidates: list[LocalCandidate]) -> str | None:
    if profile.category in {"factual", "ner"}:
        return f"measured-hard category: {profile.category}"
    if profile.category == "math":
        return "math not handled by deterministic verifier"
    if not candidates:
        return "local inference failed"
    selected = choose_local_candidate(candidates)
    if selected.finish_reason == "length":
        return "local answer was truncated"
    if not format_is_valid(prompt, selected.answer, profile.category):
        return "local answer failed format validation"
    if len(candidates) > 1:
        if not candidates_agree(candidates, profile.category):
            return "local samples disagreed"
        return None
    confidence = sum(candidate.confidence for candidate in candidates) / len(candidates)
    if confidence < profile.confidence_floor:
        return f"local confidence {confidence:.2f} below {profile.confidence_floor:.2f}"
    return None


def model_score(model: str, category: str) -> int:
    text = model.lower()
    if any(hint in text for hint in MODEL_EXCLUDE_HINTS):
        return -10_000
    score = 0
    if "instruct" in text or "chat" in text or "it" in text:
        score += 120
    if "base" in text:
        score -= 500
    for rank, hint in enumerate(CATEGORY_MODEL_HINTS[category]):
        if hint in text:
            score += 300 - rank * 25
    return score


def allowed_models() -> list[str]:
    return [model.strip() for model in os.getenv("ALLOWED_MODELS", "").split(",") if model.strip()]


def fireworks_available() -> bool:
    return bool(os.getenv("FIREWORKS_API_KEY") and os.getenv("FIREWORKS_BASE_URL") and allowed_models())


def call_fireworks(prompt: str, profile: TaskProfile, timeout: float) -> tuple[str, str, int]:
    api_key = os.getenv("FIREWORKS_API_KEY", "")
    base_url = os.getenv("FIREWORKS_BASE_URL", "")
    models = sorted(allowed_models(), key=lambda model: model_score(model, profile.category), reverse=True)
    usable = [model for model in models if model_score(model, profile.category) > -1_000]
    if not api_key or not base_url or not usable:
        raise RuntimeError("Fireworks escalation is unavailable")

    last_error: Exception | None = None
    for model in usable[:2]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"{GLOBAL_SYSTEM_PROMPT}\n{profile.guidance}"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": profile.max_tokens,
            "temperature": 0.0,
        }
        if any(hint in model.lower() for hint in ("minimax", "reason", "r1", "gpt")):
            payload["reasoning_effort"] = "low" if profile.category in {"math", "logic", "debug", "codegen"} else "none"
        try:
            response = post_json(
                chat_endpoint(base_url),
                payload,
                timeout,
                {"Authorization": f"Bearer {api_key}"},
            )
        except urllib.error.HTTPError as exc:
            if "reasoning_effort" in payload and exc.code == 400:
                payload.pop("reasoning_effort", None)
                try:
                    response = post_json(
                        chat_endpoint(base_url), payload, timeout, {"Authorization": f"Bearer {api_key}"}
                    )
                except Exception as retry_error:
                    last_error = retry_error
                    continue
            else:
                last_error = exc
                continue
        except Exception as exc:
            last_error = exc
            continue

        choices = response.get("choices") or []
        if not choices:
            last_error = RuntimeError("Fireworks returned no choices")
            continue
        answer = clean_answer(str(choices[0].get("message", {}).get("content", "")))
        if not answer:
            last_error = RuntimeError("Fireworks returned an empty answer")
            continue
        usage = response.get("usage") or {}
        return answer, model, int(usage.get("total_tokens") or 0)
    raise RuntimeError(f"Fireworks escalation failed: {last_error}")


def choose_local_candidate(candidates: list[LocalCandidate]) -> LocalCandidate:
    return max(candidates, key=lambda candidate: (candidate.confidence, -len(candidate.answer)))


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
    route_counts = {"deterministic": 0, "local": 0, "fireworks": 0, "fallback": 0}
    paid_tokens = 0

    try:
        tasks = read_tasks()
        process, server_log = start_local_server()
        results: list[dict[str, Any]] = []

        for task in tasks:
            remaining = TOTAL_RUNTIME_SECONDS - (time.monotonic() - started_at)
            if remaining < 10:
                raise TimeoutError("Not enough time remains to complete all tasks")
            prompt = task_prompt(task)
            profile = build_profile(prompt)
            deterministic = local_math_answer(prompt) if profile.category == "math" else None
            if deterministic is not None:
                route_counts["deterministic"] += 1
                logger.info("Task %s solved by deterministic math", task["task_id"])
                results.append({"task_id": task["task_id"], "answer": deterministic})
                continue

            candidates: list[LocalCandidate] = []
            sample_count = profile.samples if remaining > 140 else 1
            if remaining < 70 and profile.category in {"factual", "ner", "math"} and fireworks_available():
                sample_count = 0
            for index in range(sample_count):
                try:
                    candidates.append(
                        run_local_sample(
                            prompt,
                            profile,
                            min(TASK_TIMEOUT_SECONDS, max(20.0, remaining - 8)),
                            seed=(7, 29)[index],
                            temperature=(0.2, 0.65)[index],
                        )
                    )
                except Exception as exc:
                    logger.warning("Task %s local sample %d failed: %s", task["task_id"], index + 1, exc)
                    break

            reason = escalation_reason(prompt, profile, candidates)
            selected = choose_local_candidate(candidates) if candidates else None
            if reason and fireworks_available():
                logger.info("Task %s escalating to Fireworks: %s", task["task_id"], reason)
                try:
                    answer, model, tokens = call_fireworks(prompt, profile, min(35.0, max(10.0, remaining - 5)))
                    paid_tokens += tokens
                    route_counts["fireworks"] += 1
                    logger.info("Task %s completed by %s (%d proxy tokens)", task["task_id"], model, tokens)
                    results.append({"task_id": task["task_id"], "answer": answer})
                    continue
                except Exception as exc:
                    logger.warning("Task %s escalation failed; retaining local answer: %s", task["task_id"], exc)

            if selected is not None:
                route_counts["local"] += 1
                logger.info(
                    "Task %s accepted locally as %s (confidence %.2f, samples %d%s)",
                    task["task_id"], profile.category, selected.confidence, len(candidates),
                    f", gate override: {reason}" if reason else "",
                )
                results.append({"task_id": task["task_id"], "answer": selected.answer})
            else:
                route_counts["fallback"] += 1
                results.append({"task_id": task["task_id"], "answer": "Unable to determine the answer."})

        write_results(results)
        logger.info(
            "Wrote %d results in %.1fs | routes=%s | Fireworks proxy tokens=%d",
            len(results), time.monotonic() - started_at, route_counts, paid_tokens,
        )
        return 0
    except Exception as exc:
        logger.exception("Hybrid agent failed: %s", exc)
        return 1
    finally:
        stop_local_server(process, server_log)


if __name__ == "__main__":
    sys.exit(run())
