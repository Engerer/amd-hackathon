import os
import json
import asyncio
import logging
import sys
import re
from openai import AsyncOpenAI, APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
LOCAL_MODEL_PATH = "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Local fallback paths for testing/development
if not os.path.exists(INPUT_PATH):
    if os.path.exists("input/tasks.json"):
        INPUT_PATH = "input/tasks.json"
        OUTPUT_PATH = "output/results.json"

# Global local LLM instance
local_llm = None

def init_local_llm():
    global local_llm
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            from llama_cpp import Llama
            logger.info(f"Loading local model from {LOCAL_MODEL_PATH}...")
            # Initialize with strict memory constraints for 4GB RAM
            # n_ctx=2048 allows slightly larger context for Qwen 0.5B while staying very low memory
            local_llm = Llama(
                model_path=LOCAL_MODEL_PATH,
                n_threads=2,
                n_ctx=2048,
                n_batch=512,
                verbose=False
            )
            logger.info("Local model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
    else:
        logger.warning(f"Local model not found at {LOCAL_MODEL_PATH}. Will fallback to API for all tasks.")

def select_api_model():
    """Selects the best API model from ALLOWED_MODELS."""
    allowed_models_str = os.environ.get("ALLOWED_MODELS", "")
    if not allowed_models_str:
        logger.error("ALLOWED_MODELS environment variable is empty or not set.")
        sys.exit(1)
        
    allowed_models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]
    if not allowed_models:
        logger.error("No valid models found in ALLOWED_MODELS.")
        sys.exit(1)
        
    return allowed_models[0]

def classify_task(prompt: str) -> str:
    """Use fast regex to classify the task category for prompt optimization."""
    p_lower = prompt.lower()
    if any(k in p_lower for k in ["sentiment", "positive or negative", "negative or positive", "attitude", "label the sentiment", "classify the sentiment"]):
        return "sentiment"
    if any(k in p_lower for k in ["summarize", "summary", "summarise", "condense", "shorten", "in one sentence", "passage"]):
        return "summary"
    if any(k in p_lower for k in ["named entity", "ner", "extract the entities", "extract person", "extract organization", "extract location", "extract date", "identify person", "identify organization"]):
        return "ner"
    if any(k in p_lower for k in ["debug", "bug", "find the error", "fix the code", "correct the code", "code snippet", "incorrect implementation"]):
        return "debug"
    if any(k in p_lower for k in ["write a function", "implement a function", "write code", "python function", "javascript function", "write a python program"]):
        return "codegen"
    if any(k in p_lower for k in ["calculate", "solve the equation", "math problem", "arithmetic", "percentage", "projection", "multi-step arithmetic"]):
        return "math"
    if any(k in p_lower for k in ["logic puzzle", "deductive reasoning", "constraint-based", "riddle", "all conditions must be satisfied"]):
        return "logic"
    return "factual"

# Optimized, token-efficient system prompts
SYSTEM_PROMPTS = {
    "factual": "Provide a direct, accurate, and concise answer to the question. Do not include introductory or concluding remarks.",
    "math": "Solve the math problem step-by-step briefly, then clearly state the final answer. Keep explanations concise.",
    "sentiment": "Identify the sentiment (Positive/Negative/Neutral) and provide a one-sentence justification. Format: Sentiment: <sentiment>. Justification: <reason>.",
    "summary": "Summarize the provided text to satisfy the requested constraint (e.g. length, sentence count). Be highly concise.",
    "ner": "Extract and list the requested entities (Person, Organization, Location, Date) in a clean format.",
    "logic": "Resolve the logic puzzle step-by-step and state the final solution. Keep the reasoning clean and brief.",
    "debug": "Output ONLY the corrected code snippet. No markdown blocks unless necessary. Do not provide explanations.",
    "codegen": "Output ONLY the required code. No explanations. No comments. Be extremely concise."
}

MAX_TOKENS = {
    "factual": 150, "math": 300, "sentiment": 80, "summary": 200,
    "ner": 150, "logic": 300, 
    "debug": 350, "codegen": 500
}

def score_difficulty(user_prompt: str) -> int:
    """Use the local Qwen model to rate the difficulty of the prompt (1-10)."""
    if not local_llm:
        return 10  # Fallback to API if local model isn't available
        
    system_prompt = "You are an AI task router. Rate the complexity of the user's task on a scale from 1 to 10 based on the reasoning steps required. Output ONLY the integer."
    # Qwen chat format
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        response = local_llm(prompt, max_tokens=2, temperature=0.0, stop=["<|im_end|>"])
        text = response["choices"][0]["text"].strip()
        
        match = re.search(r'\d+', text)
        if match:
            score = int(match.group())
            return min(max(score, 1), 10)  # Clamp between 1 and 10
    except Exception as e:
        logger.warning(f"Error scoring difficulty: {e}")
        
    return 10

def run_local_inference(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Runs inference locally using Qwen2.5 (blocking call)."""
    if not local_llm:
        raise Exception("Local model not initialized.")
    
    # Qwen chat format
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    response = local_llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        stop=["<|im_end|>"]
    )
    return response["choices"][0]["text"].strip()

async def process_task(client: AsyncOpenAI, api_model: str, task: dict, semaphore: asyncio.Semaphore) -> dict:
    task_id = task.get("task_id")
    prompt = task.get("prompt", "")
    
    category = classify_task(prompt)
    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual"])
    max_tokens = MAX_TOKENS.get(category, 200)
    
    # Step 1: Score Difficulty using local LLM
    difficulty_score = await asyncio.to_thread(score_difficulty, prompt)
    
    # Step 2: Route based on difficulty threshold (>= 7 goes to API)
    use_api = difficulty_score >= 7
    
    logger.info(f"Processing task {task_id} (Category: {category}, Difficulty: {difficulty_score}, Route: {'API' if use_api else 'Local'})")
    
    if not use_api:
        # Step 3: Run local inference for easy tasks
        try:
            answer = await asyncio.to_thread(run_local_inference, system_prompt, prompt, max_tokens)
            logger.info(f"Successfully processed task {task_id} locally.")
            return {"task_id": task_id, "answer": answer}
        except Exception as e:
            logger.warning(f"Local inference failed for {task_id}: {e}. Falling back to API.")
            # Fallback to API logic below
    
    # Step 4: Run API inference for difficult tasks
    async with semaphore:
        retries = 3
        backoff = 2
        for attempt in range(retries):
            try:
                response = await client.chat.completions.create(
                    model=api_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                answer = response.choices[0].message.content.strip()
                logger.info(f"Successfully processed task {task_id} via API.")
                return {"task_id": task_id, "answer": answer}
            except APIError as e:
                logger.warning(f"API error for task {task_id} (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(f"Failed to process task {task_id} via API after {retries} attempts.")
                    return {"task_id": task_id, "answer": f"Error: Failed to process task. {str(e)}"}
            except Exception as e:
                logger.error(f"Unexpected error for task {task_id}: {e}")
                return {"task_id": task_id, "answer": f"Error: Unexpected error occurred. {str(e)}"}

async def main():
    # 1. Initialization
    init_local_llm()
    
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL")
    
    if not api_key:
        logger.error("FIREWORKS_API_KEY environment variable is missing.")
        sys.exit(1)
    if not base_url:
        logger.error("FIREWORKS_BASE_URL environment variable is missing.")
        sys.exit(1)
        
    api_model = select_api_model()
    
    # 2. Read tasks
    if not os.path.exists(INPUT_PATH):
        logger.error(f"Input file not found at {INPUT_PATH}")
        sys.exit(1)
        
    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse {INPUT_PATH}: {e}")
        sys.exit(1)
        
    if not isinstance(tasks, list):
        logger.error("Input tasks must be a list of objects.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(tasks)} tasks to process.")
    
    # 3. Process tasks
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(10)
    
    tasks_to_run = [process_task(client, api_model, task, semaphore) for task in tasks]
    results = await asyncio.gather(*tasks_to_run)
    
    # 4. Write outputs
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results written to {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Failed to write results to {OUTPUT_PATH}: {e}")
        sys.exit(1)
        
    logger.info("Agent execution completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
