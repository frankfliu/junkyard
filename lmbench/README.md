# lmbench

`lmbench` is a curl-like LLM benchmark tool written in Rust. It sends requests with a configurable
concurrency or request rate, and reports latency, TTFT, throughput, and serving metrics (OTPS, TPOT, cache hit rate by
input bucket) for streaming endpoints.

## Install

Download a prebuilt binary from [releases](https://github.com/frankfliu/junkyard/releases):

```sh
# Linux
curl -o lmbench -L https://github.com/frankfliu/junkyard/releases/download/v0.5.0/lmbench-linux-amd64
# macOS
curl -o lmbench -L https://github.com/frankfliu/junkyard/releases/download/v0.5.0/lmbench-macos-aarch64

chmod +x lmbench
```

> Build from source to get latest version:
>
> ```sh
> cargo build --release   # binary at target/release/lmbench
> # optional features: --features bedrock,harmony
> ```

## Usage

```sh
lmbench <URL>      # works like curl
lmbench -h         # print all options
```

Frequently used options:

| Option                | Description                                                                     |
|-----------------------|---------------------------------------------------------------------------------|
| `-c, --clients`       | Number of concurrent clients                                                    |
| `-N, --repeat`        | Number of requests; `0` runs each dataset record once                           |
| `--duration`          | Stop after this many seconds                                                    |
| `--request-rate`      | Poisson arrival rate (requests/second) instead of closed loop                   |
| `--dataset`           | A `.jsonl` / `.json` array file, or a directory of files, one request each      |
| `-i, --input-jq`      | jq expression that turns a dataset record into the request body                 |
| `--server`            | Response format preset: `openai`, `anthropic`, `gemini`, `TGI`                  |
| `-j, --output-jq`     | Custom jq expression to extract generated text (instead of `--server`)          |
| `-t, --tokens`        | Count tokens and report token metrics (required for serving metrics)            |
| `-o, --output`        | Write per-request results to `<dir>/output.log` and errors to `<dir>/error.log` |
| `-r, --retry-failed`  | Re-run only records missing from a previous `output.log`                        |
| `--min-output-tokens` | Minimum output tokens for a request to count towards OTPS/TPOT (default 32)     |
| `--otps-floor`        | Per-request OTPS floor reported as `otps_below_floor_ratio` (default 35)        |

### Tokenizer

Token counts use a Hugging Face tokenizer selected by `TOKENIZER_NAME`. Without it, lmbench counts
whitespace-separated words, which is not suitable for token metrics.

```sh
TOKENIZER_NAME=Qwen/Qwen3-8B lmbench ...
```

## Metrics

Stats are printed as JSON when `-N > 1` or `-v`.

### General

| Field                                         | Definition                                                               |
|-----------------------------------------------|--------------------------------------------------------------------------|
| `success_requests` / `error_requests`         | Requests that completed with a 2xx response / failed                     |
| `latency_ms`                                  | End-to-end request latency: request sent until the last byte is received |
| `ttft_ms`                                     | Time to first token, see below                                           |
| `qps`                                         | `success_requests / total_time`                                          |
| `input_tokens`, `output_tokens`               | Client-side token counts (`-t`)                                          |
| `output_tokens_per_min`                       | `output_tokens / total_time * 60`                                        |
| `server_input_tokens`, `server_output_tokens` | Token counts reported in the response `usage`                            |

Distributions report `min`, `max`, `mean`, `p50`, `p90`, `p99`. A percentile `pN` is the
`floor(count * N / 100)`-th smallest sample (0-based).

### Serving metrics

With `-t` and a streaming (`text/event-stream`) response, lmbench records the arrival time of every
response frame and reports a `serving` section. The definitions follow the Token In / Token Out
serving requirements, applied to SSE frames:

- **Token frame**: a streamed frame whose extracted text (answer *and* reasoning content) is
  non-empty. Role-only frames, usage-only frames, pings, and `[DONE]` are not token frames. A frame
  arrives when the network chunk that completes its line is received.

- **TTFT**: time from sending the request until the first token frame arrives. Connection setup,
  queueing, and prefill are included. For non-streaming responses (or when no text could be
  extracted), TTFT falls back to the time to first byte.

- **Input bucket**: requests are grouped by uncached (incremental) input tokens,
  `server prompt tokens - server cached tokens`. When the server reports no usage, the client-side
  input token count is used. Buckets are `[lower, upper)` with K = 1024:
  `0-8K`, `8K-32K`, `32K-128K`, `128K-256K`, `256K-1M`.

- **OTPS** (output tokens per second), per request:

  ```
  OTPS = (output tokens - tokens in first token frame)
         / (arrival of last token frame - arrival of first token frame)
  ```

  Output tokens are the server-reported completion tokens (including reasoning tokens) when
  available, otherwise the client-side count. TTFT is excluded.

- **TPOT** (time per output token): for each token frame after the first, the gap since the previous
  token frame is split evenly over the tokens in that frame. Percentiles are computed over all
  tokens (not per request), with a log-scale histogram at 1% resolution. OTPS and TPOT
  percentiles are computed independently; one is not the reciprocal of the other.

- **Short outputs**: requests with fewer than `--min-output-tokens` output tokens or fewer than two
  token frames are excluded from OTPS/TPOT and counted in `short_output_requests`. They still count
  towards TTFT.

- **`otps_below_floor_ratio`**: fraction of OTPS-measured requests with OTPS `<= --otps-floor`.

- **`cache_hit_rate`**: `Σ server cached tokens / Σ server prompt tokens`.

Per-frame token counts come from tokenizing each frame's text (minimum 1 token per frame), so use the
model's tokenizer via `TOKENIZER_NAME`.

Server usage fields per `--server` preset:

| Preset      | Prompt tokens                                                          | Cached tokens                               | Output tokens                        |
|-------------|------------------------------------------------------------------------|---------------------------------------------|--------------------------------------|
| `openai`    | `usage.prompt_tokens`                                                  | `usage.prompt_tokens_details.cached_tokens` | `usage.completion_tokens`            |
| `anthropic` | `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` | `cache_read_input_tokens`                   | `usage.output_tokens`                |
| `gemini`    | `usageMetadata.promptTokenCount`                                       | `usageMetadata.cachedContentTokenCount`     | `usageMetadata.candidatesTokenCount` |

OpenAI-compatible servers (vLLM, SGLang, ...) only send usage in a stream when the request sets
`"stream_options": {"include_usage": true}`. Usage repeated across frames is taken once (max), not
summed.

Example output:

```json
{
  "serving": {
    "cache_hit_rate": 0.4,
    "server_prompt_tokens": 80000,
    "server_cached_tokens": 32000,
    "all": {
      "input_tokens": "all",
      "requests": 8
    },
    "buckets": [
      {
        "input_tokens": "0-8K",
        "requests": 8,
        "short_output_requests": 0,
        "ttft_ms": {
          "min": 305,
          "max": 308,
          "mean": 307,
          "p50": 308,
          "p90": 308,
          "p99": 308
        },
        "otps": {
          "min": 42.2,
          "max": 43.88,
          "mean": 43.05,
          "p50": 43.07,
          "p90": 43.88,
          "p99": 43.88
        },
        "otps_below_floor_ratio": 0.0,
        "tpot_ms": {
          "min": 20.02,
          "max": 25.4,
          "mean": 23.23,
          "p50": 23.27,
          "p90": 25.2,
          "p99": 25.2
        }
      }
    ]
  }
}
```

Per-request values (`ttft_us`, `otps`, `uncached_input_tokens`, `server_cached_tokens`, ...) are
written to `output.log` with `-o`.

## Benchmark a dataset

### AIME 2025 (`opencompass/AIME2025`)

```sh
curl -o aime2025-I.jsonl "https://huggingface.co/api/resolve-cache/datasets/opencompass/AIME2025/a6ad95f611d72cf628a80b58bd0432ef6638f958/aime2025-I.jsonl?download=true&etag=%22553da6267ab95e09d7c590d4dc3383ccceed8cc2%22"

TOKENIZER_NAME=Qwen/Qwen3-8B lmbench "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --server openai \
  --tokens \
  --output output \
  -N 0 -c 2 \
  --input-jq '{
      "model": "qwen3.5-tiny",
      "stream": true,
      "stream_options": { "include_usage": true },
      "messages": [
        { "role": "user", "content": .question }
      ]
    }' \
  --dataset aime2025-I.jsonl
```

### GSM8K (`openai/gsm8k`)

```sh
python -c 'from datasets import load_dataset; load_dataset("openai/gsm8k", "main")["test"].to_json("gsm8k_test.jsonl", orient="records", lines=True)'

TOKENIZER_NAME=Qwen/Qwen3-8B lmbench "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --server openai \
  --tokens \
  --output output \
  -N 0 -c 30 \
  --input-jq '{
      "model": "qwen3.5-tiny",
      "messages": [
        { "role": "user", "content": .question }
      ]
    }' \
  --extra-parameters '{"stream": true, "stream_options": {"include_usage": true}}' \
  --dataset gsm8k_test.jsonl
```

`--extra-parameters` is merged into every JSON request body, so the same `--input-jq` can be used
with or without streaming.

### Load test

Run at a fixed concurrency for a period of time (the dataset is cycled), e.g. 500 concurrent clients
for one hour:

```sh
TOKENIZER_NAME=Qwen/Qwen3-8B lmbench "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --server openai --tokens --silent \
  -c 500 -N 100000000 --duration 3600 \
  --input-jq '{"model": "qwen3.5-tiny", "messages": [{"role": "user", "content": .question}]}' \
  --extra-parameters '{"stream": true, "stream_options": {"include_usage": true}}' \
  --dataset gsm8k_test.jsonl
```

Or at a target arrival rate instead of a fixed concurrency: `--request-rate 80 -c 500`
(about 4,800 RPM, with up to 500 requests in flight).

### Other servers

```sh
# Anthropic Messages API (streaming)
lmbench https://api.anthropic.com/v1/messages \
  -H "Content-Type: application/json" -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" \
  --server anthropic --tokens -N 10 \
  --input-jq '{"model": "claude-haiku-5-5", "max_tokens": 1024, "stream": true, "messages": [{"role": "user", "content": .question}]}' \
  --dataset gsm8k_test.jsonl

# Custom response format
lmbench http://localhost:8080/generate -d '{"inputs": "hello"}' -H "Content-Type: application/json" \
  --output-jq '.generated_text'
```
