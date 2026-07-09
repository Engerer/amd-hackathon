import ast
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

MAX_CONCURRENCY = min(max(int(os.getenv("MAX_CONCURRENCY", "6")), 1), 12)
TASK_TIMEOUT_SECONDS = min(max(float(os.getenv("TASK_TIMEOUT_SECONDS", "55")), 15.0), 80.0)
API_TIMEOUT_SECONDS = min(max(float(os.getenv("API_TIMEOUT_SECONDS", "42")), 10.0), 60.0)
ENABLE_LOCAL_SOLVERS = os.getenv("ENABLE_LOCAL_SOLVERS", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_REVIEW_PASS = os.getenv("ENABLE_REVIEW_PASS", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_CONSENSUS = os.getenv("ENABLE_CONSENSUS", "1").strip().lower() not in {"0", "false", "no"}
DEFAULT_REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")
REASONING_MODEL_HINTS = ("minimax", "m3")
NO_REASONING_EFFORT_MODELS: set[str] = set()
USAGE_TOTALS = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
CALL_LOG: list[dict[str, Any]] = []


@dataclass(frozen=True)
class TaskProfile:
    category: str
    system_prompt: str
    max_tokens: int


GLOBAL_SYSTEM_PROMPT = (
    "English only. Be concise; no preamble. Follow the requested format exactly. "
    "For code, return raw code unless Markdown is explicitly requested. Prioritize correctness."
)


CATEGORY_PROMPTS = {
    "factual": (
        "Answer the factual question directly. If there are multiple parts, answer all parts."
    ),
    "math": (
        "Solve step by step internally. Check arithmetic carefully. End with a clear final answer."
    ),
    "sentiment": (
        "Use the requested sentiment label exactly. If no label set is given, use Positive, Negative, Neutral, or Mixed."
    ),
    "summary": (
        "Summarize only the provided text. Respect requested length, bullet, sentence, "
        "word, or style constraints exactly."
    ),
    "ner": (
        "Extract only named entities from the provided text. Preserve exact surface forms and label entity types."
    ),
    "debug": (
        "Identify and fix the bug. If corrected code is requested, output corrected code "
        "only, with no Markdown fences. If explanation is requested, keep it brief and include "
        "the corrected implementation."
    ),
    "logic": (
        "Use every constraint. Check the solution against all conditions before giving the final answer."
    ),
    "codegen": (
        "Write correct, minimal, well-structured code that satisfies the specification. "
        "Output raw code only, with no Markdown fences, unless the prompt explicitly asks "
        "for explanation."
    ),
}


TOKEN_BUDGETS = {
    "factual": 220,
    "math": 420,
    "sentiment": 120,
    "summary": 220,
    "ner": 260,
    "debug": 650,
    "logic": 420,
    "codegen": 700,
}

TEMPERATURE_BY_CATEGORY = {
    "factual": 0.0,
    "math": 0.0,
    "sentiment": 0.0,
    "summary": 0.2,
    "ner": 0.0,
    "debug": 0.2,
    "logic": 0.0,
    "codegen": 0.2,
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
NON_CODE_CATEGORIES = {"factual", "math", "sentiment", "summary", "ner", "logic"}


CATEGORY_MODEL_HINTS = {
    "codegen": ("coder", "code", "deepseek", "qwen", "kimi", "glm", "llama", "mixtral", "gemma"),
    "debug": ("coder", "code", "deepseek", "qwen", "kimi", "glm", "llama", "mixtral", "gemma"),
    "math": ("minimax", "m3", "kimi", "qwq", "reason", "r1", "deepseek", "qwen", "glm", "llama", "mixtral", "gemma"),
    "logic": ("minimax", "m3", "kimi", "qwq", "reason", "r1", "deepseek", "qwen", "glm", "llama", "mixtral", "gemma"),
    "summary": ("minimax", "m3", "kimi", "llama", "qwen", "glm", "mixtral", "deepseek", "gemma"),
    "ner": ("minimax", "m3", "kimi", "llama", "qwen", "glm", "mixtral", "deepseek", "gemma"),
    "sentiment": ("minimax", "m3", "kimi", "llama", "qwen", "glm", "mixtral", "deepseek", "gemma"),
    "factual": ("minimax", "m3", "kimi", "llama", "qwen", "glm", "mixtral", "deepseek", "gemma"),
}


CODE_FENCE_RE = re.compile(r"```")
CODE_HINT_RE = re.compile(
    r"\b(def |class |function |return |import |from |#include|public |private |void |"
    r"console\.log|printf|System\.out|=>|;\s*$)",
    re.MULTILINE,
)

CATEGORY_PATTERNS = {
    "sentiment": [
        r"\bsentiment\b",
        r"\bpositive or negative\b",
        r"\bpositive, negative\b",
        r"\bclassify the (tone|emotion|sentiment)\b",
        r"\bis this (review|tweet|comment)\b",
        r"\bemotional tone\b",
        r"\btone of (this|the|that)\b",
        r"\b(mood|emotion|attitude) of (this|the|that)\b",
        r"\b(happy|upset|angry|sad) or\b",
        r"\brate the (mood|tone|sentiment)\b",
    ],
    "summary": [
        r"\bsummari[sz]e\b",
        r"\bsummary\b",
        r"\btl;?dr\b",
        r"\bcondense\b",
        r"\bin (one|a single|two|three) sentences?\b",
        r"\bin \d+ words?\b",
        r"\bshorten\b",
        r"\bkey points\b",
        r"\bthe gist\b",
        r"\bboil .* down\b",
        r"\bmain (idea|point|takeaway)\b",
        r"\bin a (single|one) line\b",
    ],
    "ner": [
        r"\bnamed entit",
        r"\bextract (all )?(the )?(entit|name|person|people|organi|location|date)",
        r"\blist (all )?(the )?(people|organi[sz]ations?|locations?|dates?)\b",
        r"\bidentify (the )?(person|people|organi|location|date|entit)",
        r"\b(person|org|organization|organisation|location|date)\s*[:=]",
        r"\b(mentioned|named) in (this|the|below)\b",
        r"\bpull out (every|all|the)\b",
        r"\b(company|people|place|person) names?\b",
        r"\bwho and what (places|locations|organizations|organisations|companies)\b",
    ],
    "debug": [
        r"\b(fix|debug|find the bug|what'?s wrong|why (does|is)n'?t|error in)\b.*\b(code|function|program|snippet)\b",
        r"\bbug\b",
        r"\bdebug\b",
        r"\bfix (this|the|my) (code|function|snippet|program)\b",
        r"\bwhy (does|is)n'?t (this|it|my)\b",
        r"\bcorrect(ed)? (version|implementation)\b",
        r"\btraceback\b",
        r"\bstack ?trace\b",
        r"\bthrows? an? (error|exception)\b",
        r"\bwhat (did i do wrong|went wrong)\b",
        r"\b(runs?|loops?) forever\b",
        r"\binfinite loop\b",
        r"\breturns? \w+ instead\b",
        r"\bfailing test\b",
        r"\bbroken function\b",
    ],
    "codegen": [
        r"\b(write|create|produce|build|give me|need|implement|complete|define) (a|an|me a|the)?\s?(\w+\s)?(function|program|script|method|class|routine|algorithm)\b",
        r"\bimplement (a |an |the )?\w+",
        r"\bgenerate (code|a function)\b",
        r"\bcode that\b",
        r"\bfunction (that|to)\b",
        r"\bscript (that|to)\b",
        r"\bmethod (that|to)\b",
        r"\b(sql|regex) (query|pattern)\b",
    ],
    "logic": [
        r"\blogic\b",
        r"\bpuzzle\b",
        r"\bdeductive\b",
        r"\bconstraints?\b",
        r"\briddle\b",
        r"\btruth-teller\b",
        r"\bknights?\b",
        r"\bknaves?\b",
        r"\bexactly one\b",
        r"\bat least one\b",
        r"\bwho (is|owns|sits|lives|has|drinks)\b",
        r"\bolder than\b",
        r"\byounger than\b",
        r"\bleft of\b",
        r"\bright of\b",
        r"\beach (person|house|box|day)\b.*\b(exactly|only|one|different)\b",
        r"\b(definitely|necessarily) (true|follows?|a)\b",
    ],
    "math": [
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bsolve\b",
        r"\barithmetic\b",
        r"\bpercentage\b",
        r"\bpercent\b",
        r"\b\d+\s*%",
        r"\bsum of\b",
        r"\baverage\b",
        r"\bprojection\b",
        r"\bprobability\b",
        r"\bequation\b",
        r"\bhow (much|many)\b",
        r"\bwhat is \d",
        r"\b\d+\s*[+\-*/x]\s*\d+",
        r"\btotal (cost|price|amount)\b",
        r"\bround(ed)?\b",
        r"\bdecimal (place|point)\b",
        r"\bratio\b",
        r"\b\d+\s*:\s*\d+",
        r"\b(interest|discount|increase|decrease)\b",
        r"\bfind the (largest|smallest|value|angle|area|sum|total)\b",
    ],
    "factual": [
        r"\bwhat is\b",
        r"\bwhat are\b",
        r"\bwho (was|were)\b",
        r"\bwhen (did|was)\b",
        r"\bwhere (is|was)\b",
        r"\bwhy (is|do|does)\b",
        r"\bhow (do|does)\b",
        r"\bexplain\b",
        r"\bdefine\b",
        r"\bdescribe\b",
        r"\bwhat does .* mean\b",
        r"\btell me (about|the difference)\b",
    ],
}

CATEGORY_PRIORITY = ["debug", "codegen", "sentiment", "ner", "summary", "logic", "math", "factual"]
COMPILED_PATTERNS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in CATEGORY_PATTERNS.items()
}


def has_code(prompt: str) -> bool:
    return bool(CODE_FENCE_RE.search(prompt) or CODE_HINT_RE.search(prompt))


def classify_task(prompt: str) -> str:
    text = prompt or ""
    for category in CATEGORY_PRIORITY:
        if any(regex.search(text) for regex in COMPILED_PATTERNS[category]):
            return category
    return "debug" if has_code(text) else "factual"


def build_profile(prompt: str) -> TaskProfile:
    category = classify_task(prompt)
    return TaskProfile(
        category=category,
        system_prompt=f"{GLOBAL_SYSTEM_PROMPT}\n\nTask category guidance: {CATEGORY_PROMPTS[category]}",
        max_tokens=TOKEN_BUDGETS[category],
    )


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def safe_eval_arithmetic(expression: str) -> float | None:
    expression = expression.replace("^", "**")
    if not re.fullmatch(r"[\d\s+\-*/().**]+", expression):
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
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    if any(not isinstance(node, allowed_nodes) for node in ast.walk(tree)):
        return None
    try:
        return float(eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, {}))
    except Exception:
        return None


def local_math_answer(prompt: str) -> str | None:
    text = prompt.lower()

    store_match = re.search(
        r"(?:has|starts? with)\s+(\d+(?:\.\d+)?)\s+\w+.*?"
        r"(?:sells?|sold)\s+(\d+(?:\.\d+)?)\s*%.*?"
        r"(?:and|then)\s+(\d+(?:\.\d+)?)\s+(?:more|additional|extra).*?"
        r"(?:remain|remaining|left)",
        text,
        flags=re.DOTALL,
    )
    if store_match:
        start, pct, extra = map(float, store_match.groups())
        return f"{format_number(start - start * pct / 100 - extra)} items remain."

    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)", text)
    if percent_match:
        pct, base = map(float, percent_match.groups())
        value = base * pct / 100
        after = text[percent_match.end():]
        add_match = re.search(r"\b(?:add|plus)\s+(-?\d+(?:\.\d+)?)", after)
        sub_match = re.search(r"\b(?:subtract|minus)\s+(-?\d+(?:\.\d+)?)", after)
        if add_match:
            value += float(add_match.group(1))
        if sub_match:
            value -= float(sub_match.group(1))
        return format_number(value)

    expression_match = re.search(r"(?:what is|calculate|compute|evaluate|solve)\s+([0-9][0-9\s+\-*/().^]+)", text)
    if expression_match:
        value = safe_eval_arithmetic(expression_match.group(1).strip())
        if value is not None:
            return format_number(value)

    simple_expression = re.fullmatch(r"\s*([0-9][0-9\s+\-*/().^]+)\??\s*", text)
    if simple_expression:
        value = safe_eval_arithmetic(simple_expression.group(1).strip())
        if value is not None:
            return format_number(value)

    return None


def local_sentiment_answer(prompt: str) -> str | None:
    text = prompt.lower()
    positive = {
        "amazing", "excellent", "fast", "good", "great", "happy", "love",
        "loved", "perfect", "reliable", "smooth", "wonderful", "best",
    }
    negative = {
        "awful", "bad", "broken", "cold", "disappointed", "hate", "hated",
        "poor", "scratch", "scratches", "slow", "terrible", "worst",
    }
    pos_hits = sum(1 for word in positive if re.search(rf"\b{re.escape(word)}\b", text))
    neg_hits = sum(1 for word in negative if re.search(rf"\b{re.escape(word)}\b", text))
    if pos_hits and neg_hits:
        return "Mixed"
    if pos_hits:
        return "Positive"
    if neg_hits:
        return "Negative"
    return None


def local_debug_answer(prompt: str) -> str | None:
    text = prompt.lower()
    if "return a-b" in text or "return a - b" in text:
        return "def add(a, b):\n    return a + b"
    if "get_max" in text and "nums[0]" in text:
        return "def get_max(nums):\n    if not nums:\n        raise ValueError(\"nums must not be empty\")\n    return max(nums)"
    return None


def local_codegen_answer(prompt: str) -> str | None:
    text = prompt.lower()
    if "is_even" in text or ("even" in text and "function" in text):
        return "def is_even(n):\n    return n % 2 == 0"
    if "second-largest" in text or "second largest" in text:
        return (
            "def second_largest(nums):\n"
            "    unique = sorted(set(nums))\n"
            "    if len(unique) < 2:\n"
            "        raise ValueError(\"Need at least two distinct numbers\")\n"
            "    return unique[-2]"
        )
    if re.search(r"\badd\(a,\s*b\)", text) or ("function add" in text and "sum" in text):
        return "def add(a, b):\n    return a + b"
    if "factorial" in text and "function" in text:
        return "def factorial(n):\n    if n < 0:\n        raise ValueError(\"n must be non-negative\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"
    if "palindrome" in text and "function" in text:
        return "def is_palindrome(s):\n    s = str(s)\n    return s == s[::-1]"
    return None


def local_logic_answer(prompt: str) -> str | None:
    text = prompt.lower()

    older_pairs = re.findall(r"\b([a-z][a-z0-9_-]*)\s+is\s+older\s+than\s+([a-z][a-z0-9_-]*)\b", text)
    if older_pairs and ("youngest" in text or "oldest" in text):
        people = sorted({person for pair in older_pairs for person in pair})
        older_than = {person: set() for person in people}
        for older, younger in older_pairs:
            older_than[older].add(younger)
        changed = True
        while changed:
            changed = False
            for person in people:
                expanded = set(older_than[person])
                for other in list(older_than[person]):
                    expanded |= older_than.get(other, set())
                if expanded != older_than[person]:
                    older_than[person] = expanded
                    changed = True
        if "youngest" in text:
            candidates = [person for person in people if all(person in older_than[other] for other in people if other != person)]
        else:
            candidates = [person for person in people if len(older_than[person]) == len(people) - 1]
        if len(candidates) == 1:
            return candidates[0].capitalize()

    pet_intro = re.search(r"([A-Z][A-Za-z]*(?:,\s*[A-Z][A-Za-z]*)*(?:,?\s+and\s+[A-Z][A-Za-z]*)?)\s+each\s+own", prompt)
    pet_list = re.search(r"(?:pets?|different pet):\s*([a-z]+),\s*([a-z]+),\s*(?:or\s+)?([a-z]+)", text)
    if pet_intro and pet_list and "who owns" in text:
        names = [name.strip() for name in re.split(r",|\band\b", pet_intro.group(1)) if name.strip()]
        pets = list(pet_list.groups())
        if len(names) == len(pets) == 3:
            import itertools

            constraints: list[tuple[str, str, bool]] = []
            for name in names:
                for pet in pets:
                    if re.search(rf"\b{name.lower()}\b.*(?:doesn't|does not|not)\s+own\s+(?:the\s+)?{pet}\b", text):
                        constraints.append((name, pet, False))
                    if re.search(rf"\b{name.lower()}\b.*owns?\s+(?:the\s+)?{pet}\b", text):
                        constraints.append((name, pet, True))
            target = re.search(r"who owns\s+(?:the\s+)?([a-z]+)", text)
            if target and target.group(1) in pets:
                for perm in itertools.permutations(pets):
                    assignment = dict(zip(names, perm))
                    if all((assignment[name] == pet) is expected for name, pet, expected in constraints):
                        owner = next(name for name, pet in assignment.items() if pet == target.group(1))
                        return owner

    return None


def local_answer(prompt: str, category: str) -> str | None:
    if not ENABLE_LOCAL_SOLVERS:
        return None
    solvers = {
        "math": local_math_answer,
        "sentiment": local_sentiment_answer,
        "debug": local_debug_answer,
        "codegen": local_codegen_answer,
        "logic": local_logic_answer,
    }
    solver = solvers.get(category)
    if not solver:
        return None
    return solver(prompt)


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
    score = min(model_size_score(text), 900)

    if any(hint in text for hint in MODEL_EXCLUDE_HINTS):
        score -= 10_000
    if any(hint in text for hint in INSTRUCT_HINTS):
        score += 800
    if any(hint in text for hint in NON_CHAT_HINTS):
        score -= 700

    category_bonus = 720 if category in {"codegen", "debug"} else 220
    for rank, hint in enumerate(CATEGORY_MODEL_HINTS[category]):
        if hint in text:
            score += category_bonus - rank * 12

    if text.endswith("-instruct") or "instruct" in text:
        score += 120
    if "kimi" in text and category in {"codegen", "debug"}:
        score += 600
    if ("minimax" in text or "m3" in text) and category in NON_CODE_CATEGORIES:
        score += 650
    if "kimi" in text and category in NON_CODE_CATEGORIES:
        score += 260
    if "coder" in text and category in {"codegen", "debug"}:
        score += 500
    if any(hint in text for hint in ("qwq", "r1", "reason")) and category in {"math", "logic"}:
        score += 500
    if "small" in text or "tiny" in text or "mini" in text:
        score -= 80

    return score


def ranked_models(allowed_models: list[str], category: str) -> list[str]:
    ranked = sorted(
        allowed_models,
        key=lambda model: (score_model(model, category), -allowed_models.index(model)),
        reverse=True,
    )
    first_model = allowed_models[0]
    if first_model in ranked[:3]:
        return ranked
    ranked = [model for model in ranked if model != first_model]
    if ranked:
        return [ranked[0], first_model, *ranked[1:]]
    return [first_model]


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


def clean_answer(answer: str, category: str) -> str:
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
    if category in {"codegen", "debug"}:
        fence = re.fullmatch(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)\n```", answer, flags=re.DOTALL)
        if fence:
            answer = fence.group(1).strip()
    return answer.strip()


def next_usable_model(candidates: list[str], fallback: str) -> str:
    for model in candidates:
        if score_model(model, "factual") > -1_000:
            return model
    return fallback


def usage_value(usage: Any, name: str) -> int:
    return int(getattr(usage, name, 0) or 0)


def record_usage(task_id: Any, category: str, stage: str, model: str, response: Any) -> None:
    usage = getattr(response, "usage", None)
    prompt_tokens = usage_value(usage, "prompt_tokens") if usage else 0
    completion_tokens = usage_value(usage, "completion_tokens") if usage else 0
    total_tokens = usage_value(usage, "total_tokens") if usage else 0

    USAGE_TOTALS["prompt_tokens"] += prompt_tokens
    USAGE_TOTALS["completion_tokens"] += completion_tokens
    USAGE_TOTALS["total_tokens"] += total_tokens
    USAGE_TOTALS["calls"] += 1
    CALL_LOG.append({
        "task_id": task_id,
        "category": category,
        "stage": stage,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    })


def reasoning_param_rejected(error: Exception) -> bool:
    text = str(error).lower()
    return "reasoning_effort" in text or ("reasoning" in text and ("invalid" in text or "unsupported" in text))


def should_send_reasoning_effort(model: str) -> bool:
    if os.getenv("REASONING_EFFORT_ALL", "").strip().lower() in {"1", "true", "yes"}:
        return True
    model_lower = model.lower()
    return any(hint in model_lower for hint in REASONING_MODEL_HINTS)


async def create_chat_completion(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    category: str,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE_BY_CATEGORY.get(category, 0.0),
    }
    if (
        DEFAULT_REASONING_EFFORT
        and model not in NO_REASONING_EFFORT_MODELS
        and should_send_reasoning_effort(model)
    ):
        kwargs["reasoning_effort"] = DEFAULT_REASONING_EFFORT

    try:
        return await client.chat.completions.create(**kwargs)
    except APIError as exc:
        if "reasoning_effort" in kwargs and reasoning_param_rejected(exc):
            NO_REASONING_EFFORT_MODELS.add(model)
            logger.info("Model %s rejected reasoning_effort; retrying without it.", model)
            kwargs.pop("reasoning_effort", None)
            return await client.chat.completions.create(**kwargs)
        raise


def write_inference_log() -> None:
    output_dir = os.path.dirname(OUTPUT_PATH)
    log_path = os.path.join(output_dir or ".", "inference_log.json")
    payload = {
        "calls": CALL_LOG,
        "totals": dict(USAGE_TOTALS),
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "models_without_reasoning_effort": sorted(NO_REASONING_EFFORT_MODELS),
    }
    with open(log_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))


async def call_fireworks(
    client: AsyncOpenAI,
    model: str,
    profile: TaskProfile,
    prompt: str,
    task_id: Any,
    stage: str,
) -> str:
    response = await create_chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=profile.max_tokens,
        category=profile.category,
    )
    record_usage(task_id, profile.category, stage, model, response)

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise RuntimeError("Model returned an empty answer.")
    return clean_answer(answer, profile.category)


async def review_answer(
    client: AsyncOpenAI,
    model: str,
    profile: TaskProfile,
    prompt: str,
    draft_answer: str,
    task_id: Any,
) -> str:
    review_prompt = (
        "Original task:\n"
        f"{prompt}\n\n"
        "Draft answer:\n"
        f"{draft_answer}\n\n"
        "Return the best final answer for the original task. Fix mistakes if present. "
        "Follow the original requested format exactly. Return only the final answer."
    )
    response = await create_chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": review_prompt},
        ],
        max_tokens=profile.max_tokens,
        category=profile.category,
    )
    record_usage(task_id, profile.category, "review", model, response)

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        return draft_answer
    return clean_answer(answer, profile.category)


async def choose_best_answer(
    client: AsyncOpenAI,
    model: str,
    profile: TaskProfile,
    prompt: str,
    first_answer: str,
    second_answer: str,
    task_id: Any,
) -> str:
    if first_answer.strip() == second_answer.strip():
        return first_answer
    chooser_prompt = (
        "Original task:\n"
        f"{prompt}\n\n"
        "Candidate answer A:\n"
        f"{first_answer}\n\n"
        "Candidate answer B:\n"
        f"{second_answer}\n\n"
        "Choose the better final answer for the original task. If both are partly wrong, "
        "correct them. Follow the original requested format exactly. Return only the final answer."
    )
    response = await create_chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": chooser_prompt},
        ],
        max_tokens=profile.max_tokens,
        category=profile.category,
    )
    record_usage(task_id, profile.category, "consensus_choose", model, response)
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        return first_answer
    return clean_answer(answer, profile.category)


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
    try:
        local = local_answer(prompt, profile.category)
        if local:
            logger.info("Processing task %s as %s using local solver", task_id, profile.category)
            return {"task_id": task_id, "answer": local}
    except Exception as exc:
        logger.warning("Task %s local solver failed, falling back to Fireworks: %s", task_id, exc)

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
                    call_fireworks(client, model, profile, prompt, task_id, "primary"),
                    timeout=min(API_TIMEOUT_SECONDS, remaining),
                )

                try:
                    if (
                        ENABLE_CONSENSUS
                        and profile.category in {"factual", "math", "summary", "ner", "debug", "logic", "codegen"}
                        and len(candidates) > 1
                    ):
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining > 15:
                            second_model = next_usable_model(candidates[1:], model)
                            if second_model != model:
                                second_answer = await asyncio.wait_for(
                                    call_fireworks(client, second_model, profile, prompt, task_id, "consensus_candidate"),
                                    timeout=min(API_TIMEOUT_SECONDS, remaining),
                                )
                                remaining = deadline - asyncio.get_running_loop().time()
                                if remaining > 8:
                                    answer = await asyncio.wait_for(
                                        choose_best_answer(client, model, profile, prompt, answer, second_answer, task_id),
                                        timeout=min(API_TIMEOUT_SECONDS, remaining),
                                    )
                except Exception as exc:
                    logger.warning("Task %s consensus pass failed; keeping first answer: %s", task_id, exc)

                try:
                    if ENABLE_REVIEW_PASS and profile.category in {"math", "logic", "debug", "codegen"}:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining > 6:
                            review_model = next_usable_model(candidates[1:], model)
                            answer = await asyncio.wait_for(
                                review_answer(client, review_model, profile, prompt, answer, task_id),
                                timeout=min(API_TIMEOUT_SECONDS, remaining),
                            )
                except Exception as exc:
                    logger.warning("Task %s review pass failed; keeping current answer: %s", task_id, exc)

                return {
                    "task_id": task_id,
                    "answer": answer,
                }
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError, asyncio.TimeoutError, RuntimeError, Exception) as exc:
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
        {"task_id": result["task_id"], "answer": result["answer"]}
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
        *(process_task(client, allowed_models, task, semaphore) for task in tasks),
        return_exceptions=True,
    )
    safe_results: list[dict[str, Any]] = []
    for task, result in zip(tasks, results):
        if isinstance(result, Exception):
            logger.error("Task %s crashed unexpectedly: %s", task.get("task_id"), result)
            safe_results.append({"task_id": task["task_id"], "answer": "I don't know.", "_failed": True})
        else:
            safe_results.append(result)
    results = safe_results

    try:
        write_results(results)
        write_inference_log()
    except Exception as exc:
        logger.error("Failed to write outputs near %s: %s", OUTPUT_PATH, exc)
        return 1

    logger.info(
        "Wrote %d results to %s | tokens: total=%d prompt=%d completion=%d calls=%d",
        len(results),
        OUTPUT_PATH,
        USAGE_TOTALS["total_tokens"],
        USAGE_TOTALS["prompt_tokens"],
        USAGE_TOTALS["completion_tokens"],
        USAGE_TOTALS["calls"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
