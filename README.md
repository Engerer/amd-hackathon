# AMD Developer Hackathon - Track 1 Agent

This repository contains a Track 1 container entry for the AMD Developer Hackathon.
It reads `/input/tasks.json`, sends every inference request through the Fireworks
base URL supplied by the judging harness, and writes `/output/results.json`.

## Track 1 Compliance

- Reads tasks from `/input/tasks.json` on startup.
- Writes valid JSON results to `/output/results.json` before exiting.
- Reads `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` only from
  environment variables.
- Sends all inference through the configured `FIREWORKS_BASE_URL`.
- Calls only model IDs present in `ALLOWED_MODELS`.
- Does not bundle or call local models.
- Builds a small `linux/amd64` Docker image through GitHub Actions.

## Local Contract Test

Install dependencies and run the mock Fireworks harness:

```bash
pip install -r requirements.txt
python test_agent.py
```

The test writes sample tasks, starts a local mock Fireworks API, runs `main.py`,
and verifies that every task produced a non-empty answer through an allowed model.

## Build Locally

```bash
docker buildx build --platform linux/amd64 -t amd-track1-agent .
```

## Run Locally

```bash
docker run --rm \
  -v /absolute/path/to/input:/input \
  -v /absolute/path/to/output:/output \
  -e FIREWORKS_API_KEY="provided-by-harness" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="model-a,model-b" \
  amd-track1-agent
```

## Publish

The included GitHub Actions workflow publishes a `linux/amd64` image to GHCR on
push to `main` or manual workflow dispatch:

```text
ghcr.io/engerer/amd-hackathon:latest
```

Before submitting, confirm the GHCR package is public and pullable:

```bash
docker pull ghcr.io/engerer/amd-hackathon:latest
docker inspect ghcr.io/engerer/amd-hackathon:latest
```
