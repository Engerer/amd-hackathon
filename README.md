# AMD Developer Hackathon - Track 1: General-Purpose AI Agent

This codebase implements the Track 1 submission requirements for a general-purpose AI agent running on a 2 vCPU, 4GB RAM environment with a Docker image size under 10GB.

## Features

1. **Gemma 4 E2B Priority Selection**: Scans the dynamic `ALLOWED_MODELS` list at runtime and selects the best model, prioritizing Gemma 4 E2B models (e.g. `google/gemma-4-e2b-it`).
2. **Category Classification**: Uses regex/rules to classify tasks on the fly, saving tokens and time compared to LLM-based classifiers.
3. **Optimized Prompts**: Custom-tailored system prompts for all 8 categories (Factual knowledge, Mathematical reasoning, Sentiment classification, Text summarisation, Named entity recognition, Code debugging, Logical/deductive reasoning, Code generation).
4. **Token Efficiency**: Strictly limits output tokens (`max_tokens`) per category to ensure optimal token efficiency scoring.
5. **High Concurrency**: Processes requests asynchronously using a semaphore limit of 10 to complete execution well within the 10-minute constraint.
6. **Graceful Failures**: Retries API calls with exponential backoff on transient errors.

## Project Structure

- `main.py`: Entrypoint containing agent logic.
- `requirements.txt`: Project dependencies (`openai`, `python-dotenv`).
- `Dockerfile`: Multi-stage/slim Docker container configuration.
- `test_agent.py`: Offline mock test harness.
- `.dockerignore`: Excluded files for clean Docker builds.

## How to Test and Run

### 1. Run Verification Tests Locally
Run the offline test harness to verify correct parsing, API calls, and outputs:
```bash
python test_agent.py
```
This starts a mock Fireworks server locally and processes mock tasks, generating `output/results.json`.

### 2. Build the Docker Image
To build the image locally:
```bash
docker build -t amd-track1-agent .
```

### 3. Run the Docker Container
Execute the container:
```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  -e FIREWORKS_API_KEY="your-api-key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="google/gemma-4-e2b-it,other-model" \
  amd-track1-agent
```
