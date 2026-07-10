# AMD Developer Hackathon - Track 1 Local Qwen Experiment

This branch packages Qwen3.5-2B Q4_K_M and the CPU-only llama.cpp runtime directly in the Docker image. It combines deterministic arithmetic with local Qwen inference, reads `/input/tasks.json`, and writes `/output/results.json` without making Fireworks or other external inference calls.

## Track 1 contract

- Reads tasks from `/input/tasks.json` on startup.
- Writes `[{"task_id": ..., "answer": "..."}]` to `/output/results.json`.
- Preserves task IDs and input ordering.
- Runs on `linux/amd64` with two CPU inference threads.
- Uses no Fireworks credentials and makes zero Fireworks calls.
- Solves supported arithmetic deterministically before invoking the model.
- Bundles pinned model and runtime artifacts verified by SHA-256 during the image build.

## Local model

- Model: Qwen3.5-2B instruction-tuned model
- Quantization: Q4_K_M GGUF, approximately 1.4 GB
- Runtime: llama.cpp `b9952`, CPU-only
- Context: 4096 tokens
- Server slots: 1
- CPU threads: 2

The image intentionally excludes the vision projector because Track 1 inputs are text tasks.

## Contract test

```bash
python test_agent.py
```

This test uses a local mock model server to verify input/output handling, category routing, reasoning-tag removal, and that no Fireworks environment variables are needed. It does not measure model accuracy.

## Constrained Docker test

```bash
docker build --platform linux/amd64 -t amd-track1-local-qwen .
docker run --rm --memory=4g --cpus=2 \
  -v /absolute/path/to/input:/input:ro \
  -v /absolute/path/to/output:/output \
  amd-track1-local-qwen
```

## Published image

GitHub Actions publishes the experiment as:

```text
ghcr.io/engerer/amd-hackathon:t1-local-qwen-v1
```

It also publishes an immutable `t1-<full-commit-sha>` tag. Historical experiment tags are not overwritten.
