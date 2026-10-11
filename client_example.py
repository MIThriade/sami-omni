#!/usr/bin/env python3
"""
SAMI Omni - Client Inference Example
Demonstrates multimodal and text-only inference against a running llama-server
hosting Qwen3-Omni-35B-A3B-SAMI-AsymQ and mmproj-SAMI-Omni-Vision-BF16.
"""

import sys
import time
import base64
import argparse
from pathlib import Path
import requests

def encode_image(image_path: Path) -> str:
    """Encode local image to base64 data URI."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    ext = image_path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{encoded}"

def main():
    parser = argparse.ArgumentParser(description="Query local SAMI Omni llama-server.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8080/v1/chat/completions",
                        help="Endpoint URL (default: http://127.0.0.1:8080/v1/chat/completions)")
    parser.add_argument("--prompt", type=str, default="Explain the memory bandwidth equation for sparse MoE models.",
                        help="User query prompt")
    parser.add_argument("--image", type=str, default=None,
                        help="Optional path to local image file")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum generation tokens (default: 512)")
    args = parser.parse_args()

    content = []
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[!] Error: Image file not found: {img_path}")
            sys.exit(1)
        print(f"[*] Attaching multimodal image: {img_path.name}")
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image(img_path)}
        })

    content.append({"type": "text", "text": args.prompt})

    payload = {
        "messages": [
            {"role": "user", "content": content if args.image else args.prompt}
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False
    }

    print(f"[*] Querying SAMI Omni server at {args.url}...")
    t0 = time.perf_counter()

    try:
        response = requests.post(args.url, json=payload, timeout=60)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"[!] Connection Error: Could not connect to {args.url}.")
        print("    Ensure llama-server is running on port 8080.")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Request Error: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    result = response.json()
    reply = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})

    print("\n" + "=" * 65)
    print("  💬 SAMI OMNI RESPONSE")
    print("=" * 65)
    print(reply.strip())
    print("=" * 65)

    completion_tokens = usage.get("completion_tokens", len(reply.split()))
    speed = completion_tokens / elapsed if elapsed > 0 else 0
    print(f"[*] Latency: {elapsed:.2f}s | Tokens: {completion_tokens} | Approx Speed: {speed:.1f} tok/s\n")

if __name__ == "__main__":
    main()
