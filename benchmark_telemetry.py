#!/usr/bin/env python3
"""
SAMI Omni - Benchmark Telemetry Suite
Profiles Time-To-First-Token (TTFT) and sustained token generation speed
against a local llama-server instance.
"""

import sys
import time
import json
import argparse
import requests

PROMPTS = [
    "Write a short algorithmic proof explaining why dual-channel DDR4 memory bandwidth constrains autoregressive LLM decoding.",
    "Formulate the orthogonal Procrustes problem for aligning post-ablation activation spaces.",
    "Describe the architectural difference between dense multi-head attention and sparse Mixture-of-Experts feed-forward layers."
]

def benchmark_stream(url: str, prompt: str, max_tokens: int = 128):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True
    }

    t_start = time.perf_counter()
    ttft = None
    token_count = 0

    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                data_str = decoded[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        token_count += 1
                except Exception:
                    continue

    except Exception as e:
        print(f"[!] Error during benchmark: {e}")
        return None

    total_time = time.perf_counter() - t_start
    gen_time = total_time - (ttft or 0)
    tok_per_sec = (token_count / gen_time) if gen_time > 0 else 0

    return {
        "ttft": ttft or 0,
        "tokens": token_count,
        "total_time": total_time,
        "tok_per_sec": tok_per_sec
    }

def main():
    parser = argparse.ArgumentParser(description="Profile local SAMI Omni inference telemetry.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8080/v1/chat/completions",
                        help="Server URL")
    parser.add_argument("--tokens", type=int, default=128, help="Tokens per prompt")
    args = parser.parse_args()

    print("=" * 65)
    print("  [BENCHMARK] SAMI Omni Hardware Telemetry")
    print(f"  Target: {args.url}")
    print("=" * 65)

    results = []
    for i, p in enumerate(PROMPTS, 1):
        print(f"\n[*] Running Benchmark {i}/{len(PROMPTS)}...")
        res = benchmark_stream(args.url, p, args.tokens)
        if res:
            results.append(res)
            print(f"    TTFT: {res['ttft']:.2f}s | Tokens: {res['tokens']} | Speed: {res['tok_per_sec']:.1f} tok/s")

    if not results:
        print("\n[!] No successful benchmark runs. Is the server running?")
        sys.exit(1)

    avg_ttft = sum(r["ttft"] for r in results) / len(results)
    avg_speed = sum(r["tok_per_sec"] for r in results) / len(results)

    print("\n" + "=" * 65)
    print("  [SUMMARY] Latency & Throughput Metrics")
    print("=" * 65)
    print(f"  Average Time-To-First-Token (TTFT): {avg_ttft:.2f} seconds")
    print(f"  Average Generation Throughput:       {avg_speed:.1f} tokens/second")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
