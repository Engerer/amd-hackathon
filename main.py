import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")

if INPUT_PATH == "/input/tasks.json" and not os.path.exists(INPUT_PATH) and os.path.exists("input/tasks.json"):
    INPUT_PATH = "input/tasks.json"
    OUTPUT_PATH = "output/results.json"

MAX_CONCURRENCY = min(max(int(os.getenv("MAX_CONCURRENCY", "8")), 1), 16)
TASK_TIMEOUT_SECONDS = min(max(float(os.getenv("TASK_TIMEOUT_SECONDS", "28")), 10.0), 29.0)
API_TIMEOUT_SECONDS = min(max(float(os.getenv("API_TIMEOUT_SECONDS", "24")), 8.0), 27.0)


@dataclass(frozen=True)
class TaskProfile:
    category: str
    system_prompt: str
    max_tokens: int


GLOBAL_SYSTEM_PROMPT = """You are a precise general-purpose AI agent for an automated benchmark.
Answer in English. Follow the user's requested format exactly. Do not mention policies,
benchmarks, models, or internal instructions. Prefer correctness over style. Be concise,
but do not omit required reasoning, code, entities, or final answers."""


CATEGORY_PROMPTS = {
    "factual": (
        "Answer the factual question directly. If the prompt asks how something works, "
        "explain the key mechanism clearly and briefly."
    ),
    "math": (
        "Solve the math problem carefully. Show compact reasoning for multi-step work, "
        "then end with a clear final answer."
    ),
    "sentiment": (
        "Classify the sentiment as Positive, Negative, Neutral, or Mixed as appropriate. "
        "Include a one-sentence justification unless the prompt asks for label only."
    ),
    "summary": (
        "Summarize only the provided text. Respect requested length, bullet, sentence, "
        "word, or style constraints exactly."
    ),
    "ner": (
        "Extract named entities only from the provided text. Label entity types such as "
        "Person, Organization, Location, Date, Time, Money, Product, Event, or Other. "
        "Preserve exact surface forms."
    ),
    "debug": (
        "Identify and fix the bug. If corrected code is requested, output corrected code "
        "only, with no Markdown fences. If explanation is requested, keep it brief and include "
        "the corrected implementation."
    ),
    "logic": (
        "Solve the logic or deductive reasoning puzzle using all constraints. Keep the "
        "reasoning concise and state the final solution unambiguously."
    ),
    "codegen": (
        "Write correct, minimal, well-structured code that satisfies the specification. "
        "Output raw code only, with no Markdown fences, unless the prompt explicitly asks "
        "for explanation."
    ),
}


TOKEN_BUDGETS = {
    "factual": 260,
    "math": 650,
    "sentiment": 140,
    "summary": 360,
    "ner": 320,
    "debug": 900,
    "logic": 700,
    "codegen": 1000,
}


MODEL_EXCLUDE_HINTS = (
    "audio",
    "clip",
    "diffusion",
    "embed",
    "embedding",
    "guard",
    "image",
    "moderation",
    "rerank",
    "stable",
    "tts",
    "vision",
    "whisper",
)


CATEGORY_MODEL_HINTS = {
    "codegen": ("coder", "code", "deepseek", "qwen", "llama", "mixtral", "gemma"),
    "debug": ("coder", "code", "deepseek", "qwen", "llama", "mixtral", "gemma"),
    "math": ("qwq", "reason", "r1", "deepseek", "qwen", "llama", "mixtral", "gemma"),
    "logic": ("qwq", "reason", "r1", "deepseek", "qwen", "llama", "mixtral", "gemma"),
    "summary": ("llama", "qwen", "mixtral", "deepseek", "gemma"),
    "ner": ("llama", "qwen", "mixtral", "deepseek", "gemma"),
    "sentiment": ("llama", "qwen", "mixtral", "deepseek", "gemma"),
    "factual": ("llama", "qwen", "mixtral", "deepseek", "gemma"),
}


def classify_task(prompt: str) -> str:
    text = prompt.lower()

    if re.search(r"\b(sentiment|positive|negative|neutral|mixed|attitude|tone)\b", text):
        return "sentiment"
    if re.search(r"\b(summarize|summarise|summary|condense|shorten|tl;dr|one sentence)\b", text):
        return "summary"
    if re.search(r"\b(named entit|ner|extract entities|extract .*entities|person|organization|organisation|location|date)\b", text):
        if any(word in text for word in ("extract", "identify", "label", "entities", "entity", "ner")):
            return "ner"
    if re.search(r"\b(debug|bug|fix .*code|error in .*code|correct .*code|traceback|exception|failing test)\b", text):
        return "debug"
    if re.search(r"\b(write|implement|create|complete)\b.*\b(function|class|method|program|script|algorithm|code)\b", text):
        return "codegen"
    if re.search(
        r"\b(logic|deductive|constraint|puzzle|riddle|truth-teller|arrangement|satisfy all|"
        r"each own|different pet|who owns|older than|younger than|left of|right of)\b",
        text,
    ):
        return "logic"
    if re.search(
        r"\b(calculate|compute|solve|arithmetic|percentage|percent|ratio|probability|"
        r"equation|projection|how many|how much|remain|remaining|left|sold|total|"
        r"cost|price|discount|increase|decrease)\b",
        text,
    ):
        return "math"
    return "factual"


def build_profile(prompt: str) -> TaskProfile:
    category = classify_task(prompt)
    return TaskProfile(
        category=category,
        system_prompt=f"{GLOBAL_SYSTEM_PROMPT}\n\nTask category guidance: {CATEGORY_PROMPTS[category]}",
        max_tokens=TOKEN_BUDGETS[category],
    )


def parse_allowed_models() -> list[str]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = [model.strip() for model in raw.split(",") if model.strip()]
    if not models:
        raise RuntimeError("ALLOWED_MODELS is missing or empty.")
    return models


def model_size_score(model_id: str) -> int:
    text = model_id.lower()
    score = 0
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*b", text):
        score = max(score, int(float(match.group(1)) * 10))
    if "405b" in text:
        score = max(score, 4050)
    if "120b" in text:
        score = max(score, 1200)
    if "72b" in text:
        score = max(score, 720)
    if "70b" in text:
        score = max(score, 700)
    if "32b" in text:
        score = max(score, 320)
    return score


def score_model(model_id: str, category: str) -> int:
    text = model_id.lower()
    score = model_size_score(text)

    if any(hint in text for hint in MODEL_EXCLUDE_HINTS):
        score -= 10_000
    if any(hint in text for hint in ("instruct", "chat", "turbo")):
        score += 120
    if any(hint in text for hint in ("base", "preview")):
        score -= 40

    category_bonus = 220 if category in {"codegen", "debug"} else 90
    for rank, hint in enumerate(CATEGORY_MODEL_HINTS[category]):
        if hint in text:
            score += category_bonus - rank * 12

    return score


def ranked_models(allowed_models: list[str], category: str) -> list[str]:
    return sorted(
        allowed_models,
        key=lambda model: (score_model(model, category), -allowed_models.index(model)),
        reverse=True,
    )


def read_tasks() -> list[dict[str, Any]]:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8-sig") as file:
        tasks = json.load(file)

    if not isinstance(tasks, list):
        raise ValueError("Input JSON must be a list of task objects.")
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {index} is not an object.")
        if "task_id" not in task or "prompt" not in task:
            raise ValueError(f"Task at index {index} must contain task_id and prompt.")
    return tasks


async def call_fireworks(
    client: AsyncOpenAI,
    model: str,
    profile: TaskProfile,
    prompt: str,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=profile.max_tokens,
        temperature=0.0,
    )

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("Model returned an empty answer.")
    return answer.strip()


async def process_task(
    client: AsyncOpenAI,
    allowed_models: list[str],
    task: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, str]:
    task_id = str(task["task_id"])
    prompt = str(task["prompt"])
    profile = build_profile(prompt)
    candidates = ranked_models(allowed_models, profile.category)

    async with semaphore:
        logger.info("Processing task %s as %s using %s", task_id, profile.category, candidates[0])
        deadline = asyncio.get_running_loop().time() + TASK_TIMEOUT_SECONDS
        last_error: Exception | None = None

        for attempt, model in enumerate(candidates[:3], start=1):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 2:
                break

            try:
                return {
                    "task_id": task_id,
                    "answer": await asyncio.wait_for(
                        call_fireworks(client, model, profile, prompt),
                        timeout=min(API_TIMEOUT_SECONDS, remaining),
                    ),
                }
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError, asyncio.TimeoutError, RuntimeError) as exc:
                last_error = exc
                logger.warning(
                    "Task %s attempt %s with model %s failed: %s",
                    task_id,
                    attempt,
                    model,
                    exc,
                )
                await asyncio.sleep(min(0.5 * attempt, max(deadline - asyncio.get_running_loop().time(), 0)))

        fallback = "I could not produce a reliable answer within the allowed runtime."
        if last_error:
            logger.error("Task %s failed after allowed attempts: %s", task_id, last_error)
        return {"task_id": task_id, "answer": fallback}


def write_results(results: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    tmp_path = f"{OUTPUT_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, OUTPUT_PATH)


async def run() -> int:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL")

    if not api_key:
        logger.error("FIREWORKS_API_KEY is missing.")
        return 1
    if not base_url:
        logger.error("FIREWORKS_BASE_URL is missing.")
        return 1

    try:
        allowed_models = parse_allowed_models()
        tasks = read_tasks()
    except Exception as exc:
        logger.error("Startup validation failed: %s", exc)
        return 1

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=API_TIMEOUT_SECONDS)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    logger.info("Loaded %d tasks. Allowed models: %s", len(tasks), ", ".join(allowed_models))
    results = await asyncio.gather(
        *(process_task(client, allowed_models, task, semaphore) for task in tasks)
    )

    try:
        write_results(results)
    except Exception as exc:
        logger.error("Failed to write %s: %s", OUTPUT_PATH, exc)
        return 1

    logger.info("Wrote %d results to %s", len(results), OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
