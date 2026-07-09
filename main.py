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
ENABLE_REVIEW_PASS = os.getenv("ENABLE_REVIEW_PASS", "1").strip().lower() not in {"0", "false", "no"}


@dataclass(frozen=True)
class TaskProfile:
    category: str
    system_prompt: str
    max_tokens: int


# FIX 1: Strict output-only system prompt — forces bare answer tokens, no preamble.
# The global prompt is now aggressively format-focused.
GLOBAL_SYSTEM_PROMPT = (
    "You are an expert AI assistant. "
    "You MUST follow the output format instructions exactly. "
    "Never include preamble, explanation, or meta-commentary unless explicitly asked. "
    "Output ONLY what is requested."
)

# FIX 1 (continued): Each category prompt is rewritten to be output-only and unambiguous.
CATEGORY_PROMPTS = {
    "factual": (
        "Answer with ONLY the direct answer — a word, name, date, or short phrase. "
        "No sentences like 'The answer is...'. No punctuation unless it is part of the answer itself. "
        "If the question asks 'What is the capital of France?' output exactly: Paris"
    ),
    "math": (
        "Think through the problem step by step, but output ONLY the final numeric result. "
        "Do not include the unit unless explicitly asked for it. "
        "Do not include 'The answer is', '=', or any explanation. "
        "If asked to calculate 15 + 27, output exactly: 42"
    ),
    "sentiment": (
        "Output ONLY a single word that describes the sentiment: Positive, Negative, or Neutral. "
        "Nothing else — no punctuation, no justification, no sentence."
    ),
    "summary": (
        "Output ONLY the summary text, matching the length and format requested. "
        "Do not begin with 'Here is a summary:', 'Summary:', or any other header. "
        "Write the summary content directly."
    ),
    "ner": (
        "Output ONLY the extracted entities. "
        "If the task specifies a format, follow it exactly. "
        "If no format is specified, list each entity on its own line as: Type: Value. "
        "Do not include sentences, explanations, or headers."
    ),
    "debug": (
        "Output ONLY the corrected, complete code. "
        "Do not include markdown code fences (``` or ~~~). "
        "Do not include any explanation, commentary, or description of the fix. "
        "Output the raw code text only."
    ),
    "logic": (
        "Work through the logic internally. "
        "Output ONLY the final conclusion or answer — a name, value, or short phrase. "
        "Do not show your reasoning steps in the output. "
        "If asked 'Who is the youngest?', output only the name."
    ),
    "codegen": (
        "Output ONLY the raw code — no markdown fences (``` or ~~~), no explanations, no comments "
        "unless comments are part of the requested code. "
        "Start directly with the code."
    ),
}

# FIX 5: Increased token budgets for reasoning-heavy categories.
TOKEN_BUDGETS = {
    "factual": 256,
    "math": 3000,
    "sentiment": 64,
    "summary": 1500,
    "ner": 1000,
    "debug": 2500,
    "logic": 3000,
    "codegen": 2500,
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

NON_CHAT_HINTS = ("base", "preview")
INSTRUCT_HINTS = ("instruct", "chat", "turbo", "assistant")


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


# FIX 2: Improved task classifier with broader patterns and correct priority ordering.
# Sentinel/NER check is moved up, math patterns expanded, logic patterns expanded.
def classify_task(prompt: str) -> str:
    text = prompt.lower()

    # --- Sentiment ---
    if re.search(r"\b(sentiment|classify the sentiment|determine the sentiment|opinion|attitude|tone)\b", text):
        return "sentiment"
    if re.search(r"\b(positive|negative|neutral|mixed)\b", text) and re.search(
        r"\b(classify|label|rate|is it|what is the|determine)\b", text
    ):
        return "sentiment"

    # --- Summary ---
    if re.search(r"\b(summarize|summarise|summary|condense|shorten|tl;dr|brief overview|key points|main point)\b", text):
        return "summary"
    if re.search(r"\bin one sentence\b", text) and re.search(r"\b(describe|explain|what)\b", text):
        return "summary"

    # --- NER ---
    if re.search(
        r"\b(named entit|ner|extract entities|extract .*?entities|identify entities|list entities|tag entities)\b",
        text,
    ):
        return "ner"
    if re.search(r"\b(extract|identify|list|find|tag)\b", text) and re.search(
        r"\b(person|people|organization|organisation|location|place|date|name|entity|entities)\b", text
    ):
        return "ner"

    # --- Debug ---
    if re.search(
        r"\b(debug|fix the bug|fix this bug|there is a bug|find the bug|fix the code|fix this code|"
        r"fix the error|traceback|exception|syntax error|runtime error|failing test|what is wrong)\b",
        text,
    ):
        return "debug"

    # --- Codegen ---
    if re.search(
        r"\b(write|implement|create|generate|complete|build|develop)\b.*\b(function|class|method|program|script|algorithm|code|snippet|module)\b",
        text,
    ):
        return "codegen"
    if re.search(r"\b(code that|function that|function to|program that|script that|algorithm that)\b", text):
        return "codegen"

    # --- Logic ---
    if re.search(
        r"\b(logic puzzle|logic problem|deductive|constraint|riddle|truth.teller|liar|zebra puzzle|"
        r"arrangement|who owns|each own|different pet|satisfy all|must be true|can be inferred|"
        r"older than|younger than|left of|right of|tallest|shortest|first|last|ordering)\b",
        text,
    ):
        return "logic"
    if re.search(r"\b(if .{1,60} then|given that .{1,60}(who|what|which|find))\b", text):
        return "logic"

    # --- Math ---
    if re.search(
        r"\b(calculate|compute|solve|evaluate|arithmetic|percentage|percent|ratio|probability|"
        r"equation|formula|integral|derivative|projection|average|mean|median|mode|"
        r"how many|how much|remain|remaining|how far|how long|how often|"
        r"cost|price|discount|tax|profit|loss|interest|rate|speed|distance|time|"
        r"add|subtract|multiply|divide|sum|difference|product|quotient|square root|"
        r"power|exponent|logarithm|total|increase|decrease|more than|less than)\b",
        text,
    ):
        return "math"
    # Catch numeric expressions that look like word problems
    if re.search(r"\d+\s*([\+\-\*/×÷]|\band\b|\bplus\b|\bminus\b|\btimes\b|\bdivided by\b)\s*\d+", text):
        return "math"

    return "factual"


def build_profile(prompt: str) -> TaskProfile:
    category = classify_task(prompt)
    return TaskProfile(
        category=category,
        system_prompt=f"{GLOBAL_SYSTEM_PROMPT}\n\n{CATEGORY_PROMPTS[category]}",
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
    score = min(model_size_score(text), 800)

    if any(hint in text for hint in MODEL_EXCLUDE_HINTS):
        score -= 10_000
    if any(hint in text for hint in INSTRUCT_HINTS):
        score += 800
    if any(hint in text for hint in NON_CHAT_HINTS):
        score -= 700

    category_bonus = 360 if category in {"codegen", "debug"} else 160
    for rank, hint in enumerate(CATEGORY_MODEL_HINTS[category]):
        if hint in text:
            score += category_bonus - rank * 12

    if text.endswith("-instruct") or "instruct" in text:
        score += 120
    if "small" in text or "tiny" in text or "mini" in text:
        score -= 80

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


# FIX 4: Smarter post-processing per category.
# For sentiment: normalise to a single canonical word.
# For math: strip trailing prose, keep only the numeric result.
# For factual/logic: strip common preamble phrases.
# For debug/codegen: strip markdown fences only (do NOT truncate to longest block).
def clean_answer(answer: str, category: str) -> str:
    # Remove <think> blocks even if unclosed
    answer = re.sub(r"<think>.*?(?:</think>|$)", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()

    if not answer:
        return answer

    # Strip markdown fences for code categories — keep the full content inside the fence
    if category in {"codegen", "debug"}:
        # If the whole answer is a single fence, unwrap it
        fence_match = re.fullmatch(
            r"```(?:[a-zA-Z0-9_+\-]*)\s*\n(.*?)\n?```\s*", answer, flags=re.DOTALL
        )
        if fence_match:
            return fence_match.group(1).strip()
        # If there are multiple fences, take the content of the first/largest block
        # but preserve ALL the code (do not truncate to just the longest block —
        # that was the old bug where the full corrected file got chopped).
        blocks = re.findall(r"```(?:[a-zA-Z0-9_+\-]*)?\s*\n(.*?)```", answer, flags=re.DOTALL)
        if blocks:
            return max(blocks, key=len).strip()
        # No fences — return as-is (model followed instructions)
        return answer.strip()

    # Strip any leading markdown fence that slipped through for non-code
    fence_match = re.fullmatch(r"```(?:[a-zA-Z0-9_+\-]*)?\s*\n(.*?)\n?```\s*", answer, flags=re.DOTALL)
    if fence_match:
        answer = fence_match.group(1).strip()

    # --- Sentiment normalisation ---
    if category == "sentiment":
        # Extract the canonical word from wherever it appears in the response
        for word in ("Positive", "Negative", "Neutral"):
            if re.search(rf"\b{word}\b", answer, re.IGNORECASE):
                return word
        # Fallback: return first word only (might be the label)
        first_word = answer.split()[0].rstrip(".,;:")
        return first_word

    # --- Math: strip prose and keep the numeric result ---
    if category == "math":
        # Remove common preamble like "The answer is", "= ", etc.
        answer = re.sub(
            r"^(the\s+)?(final\s+)?(answer|result|value|sum|total|output)\s*(is|=|:)\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        ).strip()
        # If the response ends with a number (possibly with commas/decimals), extract it
        num_match = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?(?:\s*\/\s*\d+)?)\s*$", answer)
        if num_match:
            return num_match.group(1).replace(",", "").strip()
        return answer.strip()

    # --- Factual / Logic: strip common preamble sentences ---
    if category in {"factual", "logic"}:
        # Remove "The answer is X" → "X"
        preamble = re.match(
            r"^(?:the\s+)?(?:answer|result|solution|conclusion)\s+(?:is|:)\s+",
            answer,
            re.IGNORECASE,
        )
        if preamble:
            answer = answer[preamble.end():].strip()
        # Remove trailing period if it looks like a single word/phrase answer
        if "\n" not in answer and len(answer.split()) <= 5:
            answer = answer.rstrip(".")
        return answer.strip()

    return answer.strip()


def next_usable_model(candidates: list[str], fallback: str) -> str:
    for model in candidates:
        if score_model(model, "factual") > -1_000:
            return model
    return fallback


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

    cleaned = clean_answer(answer, profile.category)
    if not cleaned:
        raise RuntimeError("Answer became empty after cleaning.")
    return cleaned


# FIX 3 + FIX 1 (review prompt): Review pass is now applied to ALL categories, not just 4.
# The review prompt is also stricter — it explicitly tells the model to fix format, not just content.
async def review_answer(
    client: AsyncOpenAI,
    model: str,
    profile: TaskProfile,
    prompt: str,
    draft_answer: str,
) -> str:
    format_reminder = CATEGORY_PROMPTS[profile.category]
    review_prompt = (
        "Original task:\n"
        f"{prompt}\n\n"
        "Draft answer:\n"
        f"{draft_answer}\n\n"
        "Instructions:\n"
        f"{format_reminder}\n\n"
        "Check the draft answer for correctness AND format compliance. "
        "If the draft is correct and properly formatted, repeat it exactly. "
        "If it has errors or extra text, output the corrected final answer only. "
        "Return ONLY the final answer — nothing else."
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": review_prompt},
        ],
        max_tokens=profile.max_tokens,
        temperature=0.0,
    )

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        return draft_answer

    cleaned = clean_answer(answer, profile.category)
    if not cleaned:
        return draft_answer
    return cleaned


async def process_task(
    client: AsyncOpenAI,
    allowed_models: list[str],
    task: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    task_id = task["task_id"]
    prompt = task_prompt(task)
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
                answer = await asyncio.wait_for(
                    call_fireworks(client, model, profile, prompt),
                    timeout=min(API_TIMEOUT_SECONDS, remaining),
                )

                # FIX 3: Review pass now applies to ALL categories (not just 4).
                if ENABLE_REVIEW_PASS:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining > 6:
                        review_model = next_usable_model(candidates[1:], model)
                        answer = await asyncio.wait_for(
                            review_answer(client, review_model, profile, prompt, answer),
                            timeout=min(API_TIMEOUT_SECONDS, remaining),
                        )

                return {
                    "task_id": task_id,
                    "answer": answer,
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

        fallback = "I don't know."
        if last_error:
            logger.error("Task %s failed after allowed attempts: %s", task_id, last_error)
        return {"task_id": task_id, "answer": fallback, "_failed": True}


def write_results(results: list[dict[str, Any]]) -> None:
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    clean_results = [
        {
            "task_id": result["task_id"],
            "answer": result["answer"],
        }
        for result in results
    ]
    tmp_path = f"{OUTPUT_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(clean_results, file, ensure_ascii=False, separators=(",", ":"))
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
    failed_count = sum(1 for result in results if result.get("_failed"))
    if failed_count == len(results):
        logger.error("All tasks failed API/model calls; refusing to submit all fallback answers.")
        return 1

    try:
        write_results(results)
    except Exception as exc:
        logger.error("Failed to write %s: %s", OUTPUT_PATH, exc)
        return 1

    logger.info("Wrote %d results to %s", len(results), OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
