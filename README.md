# AMD Developer Hackathon - Track 1 Hybrid Agent

This image combines deterministic verification, bundled Qwen3.5-2B inference, confidence-aware consensus, and selective Fireworks escalation for Track 1.

## Routing strategy

1. Supported arithmetic is solved deterministically at zero token cost.
2. Every other task is attempted by the bundled Qwen3.5-2B Q4_K_M model.
3. Sentiment and logic use two independent local samples when the runtime budget permits.
4. The gate checks sample agreement, generated-token log probabilities, answer format, truncation, and category risk.
5. Factual knowledge, named entity recognition, unsupported math, and uncertain local answers escalate through the injected `FIREWORKS_BASE_URL`.
6. If escalation fails, the best valid local answer is retained instead of crashing.

The gate uses transparent heuristic thresholds. It is not described as statistically calibrated because no hidden evaluation data or answer cache is used.

## Track 1 contract

- Reads `/input/tasks.json` on startup.
- Writes only `task_id` and `answer` fields to `/output/results.json`.
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from the environment.
- Calls only models listed in `ALLOWED_MODELS` and routes every paid call through `FIREWORKS_BASE_URL`.
- Bundles Qwen3.5-2B and llama.cpp; no model download occurs during evaluation.
- Runs on `linux/amd64` with two local inference threads.
- Uses no personal API key and contains no cached evaluation answers.

## Tests

```bash
python test_agent.py
```

The contract test uses local mock servers to verify deterministic routing, consensus calls, selective escalation, allowed-model enforcement, fallback behavior, and the required result schema.

For the constrained container test:

```bash
docker build --platform linux/amd64 -t amd-track1-hybrid .
docker run --rm --memory=4g --cpus=2 \
  -v /absolute/path/to/input:/input:ro \
  -v /absolute/path/to/output:/output \
  -e FIREWORKS_API_KEY="provided-by-harness" \
  -e FIREWORKS_BASE_URL="provided-by-harness" \
  -e ALLOWED_MODELS="provided-by-harness" \
  amd-track1-hybrid
```

## Published image

```text
ghcr.io/engerer/amd-hackathon:t1-hybrid-v1
```

The workflow also publishes an immutable `t1-<full-commit-sha>` tag. The previous `t1-local-qwen-v1` image remains available as the zero-token baseline.
