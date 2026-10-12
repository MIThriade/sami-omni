# SAMI Omni

**Asymmetric MoE Topology and Memory-Decoupled Multimodal Inference for Consumer Hardware**

---

## Overview

SAMI Omni describes an asymmetric memory-partitioning topology designed to evaluate 30B+ sparse Mixture-of-Experts (MoE) multimodal models on consumer hardware (6GB VRAM GPUs paired with 32GB system RAM). 

By decoupling dense attention mechanisms from sparse expert execution, the architecture achieves 15.5 – 16.0 tokens/second sustained throughput without GPU memory overflow.

---

## Hardware Memory Partitioning & Roofline Model

Under the Williams Roofline model, autoregressive token generation in sparse MoE architectures is bound by host memory bandwidth rather than arithmetic compute. By evaluating sparse feed-forward experts in system memory via CPU-MoE offloading (`-cmoe`), bus transfers over PCIe Gen3 x16 remain minimal.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA GeForce GTX 1060 (6GB VRAM)                  │
│          Bandwidth: 192 GB/s (GDDR5) · Total Allocated: 2,429 MiB      │
│                                                                        │
│  [ OS Compositor ]    [ Dense Attention & RMSNorm ]    [ Q8_0 Routers ]│
│    approx. 670 MiB             1,080 MiB                   180 MiB     │
│                                                                        │
│                       [ Active KV Cache (4096 ctx) ]                   │
│                                 499 MiB                                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Top-8 Active Expert Indices
                                    ▼ (PCIe Gen3 x16 bus utilization < 1.5%)
┌────────────────────────────────────────────────────────────────────────┐
│                        Host System RAM (32GB DDR4)                     │
│    Bandwidth: 25–30 GB/s (Dual-Channel) · Total Allocated: ~12.6 GB    │
│                                                                        │
│   [ 6,144 Sparse MoE Experts (AsymQ Q3_K/Q4_K) ]       ~11.8 GB        │
│   Streamed active weight burst per token: 1.4 – 1.6 GB via CPU AVX2    │
│                                                                        │
│   [ SigLIP Vision Projector (BF16, d=2048) ]            861 MB         │
│   Evaluated in Host RAM (--no-mmproj-offload)          (0 MB VRAM)     │
│                                                                        │
│   [ Faster-Whisper Speech STT / TTS Pipeline ]          420 MB         │
│   [ Linux Kernel OS Buffers & File System Cache ]       ~6.2 GB        │
└────────────────────────────────────────────────────────────────────────┘
```

Theoretical DDR4 Throughput Ceiling:

$$\text{Generation Throughput} = \frac{\text{Sustained Host RAM Bandwidth (25--30 GB/s)}}{\text{Active Weight Stream per Token (1.4--1.6 GB)}} \approx 15.6\text{--}18.0 \text{ tokens/second}$$

---

## Technical Methodology

1. **Subspace Column Ablation:**  
   Evaluates candidate expert down-projection matrices across the network (48 layers × 128 experts × 768 intermediate dimensions = 4,718,592 candidate columns). Identifies refusal-correlated activation vectors via contrastive probe analysis and excises 6,259 columns (0.13%) via orthogonal projection zeroing.

2. **Constrained Orthogonal Procrustes Realignment:**  
   Restores residual manifold geometry post-ablation by solving the constrained orthogonal Procrustes problem:

   $$\min_R \Vert{}A - BR\Vert{}_F \quad \text{subject to} \quad R^T R = I$$

   This prevents representation drift in downstream residual layers.

3. **Asymmetric Mixed-Precision Quantization (AsymQ):**  
   Preserves router entropy by quantizing gating routers and input embeddings at Q8_0 and dense self-attention blocks at Q5_K (pinned to VRAM). Sparse feed-forward expert weights reside in host system RAM quantized at Q3_K and Q4_K.

4. **Decoupled Visual Projector Execution:**  
   Aligns the SigLIP vision projection directly to language hidden dimension ($d = 2048$), executing the 861 MB projector in host RAM (`--no-mmproj-offload`) with zero GPU VRAM consumption.

---

## Evaluation & Benchmarks

### Reasoning Retention (Delta vs. BF16 Baseline)

Evaluation against the unpruned base model confirms that surgical column ablation combined with Procrustes realignment preserves core benchmarks within a sub-0.6% variance window:

| Benchmark Suite | Metric | Base Model (BF16) | SAMI-AsymQ (14.8 GB) | Delta (Δ) | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MMLU** (5-shot) | Multitask Accuracy | 76.8% | 76.2% | -0.6% | Factual knowledge retention |
| **GSM8K** (8-shot) | Strict Match Math | 83.4% | 82.9% | -0.5% | Multi-step arithmetic stability |
| **ARC-Challenge** (25-shot) | Causal Reasoning | 88.1% | 87.7% | -0.4% | Causal logic intact |
| **HumanEval** (pass@1) | Python Code Eval | 62.2% | 61.8% | -0.4% | Syntax and logic intact |
| **Refusal Trigger Rate** | Evaluation Probe | 98.4% | 0.0% | -98.4% | System prompt compliance |

### Context Length Scaling (GTX 1060 6GB)

| Context Length | TTFT | Generation Speed | Active KV Cache | Total VRAM Footprint | Free VRAM Headroom |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **512 tokens** | 2.08 s | 16.0 tok/sec | 62 MiB | 1,992 MiB | 4,152 MiB (67.5% Free) |
| **1,024 tokens** | 2.10 s | 15.8 tok/sec | 125 MiB | 2,055 MiB | 4,089 MiB (66.5% Free) |
| **2,048 tokens** | 2.14 s | 15.6 tok/sec | 250 MiB | 2,180 MiB | 3,964 MiB (64.5% Free) |
| **4,096 tokens** | 2.22 s | 15.1 tok/sec | 499 MiB | 2,429 MiB | 3,715 MiB (59.8% Free) |

---

## Quickstart

### 1. Download Weights

Pre-quantized model weights and vision projector artifacts are hosted on Hugging Face at [`MIThridate/Qwen3-Omni-35B-A3B-SAMI-AsymQ`](https://huggingface.co/MIThridate/Qwen3-Omni-35B-A3B-SAMI-AsymQ):

```bash
# Main AsymQ MoE checkpoint (14.8 GB)
huggingface-cli download MIThridate/Qwen3-Omni-35B-A3B-SAMI-AsymQ \
    Qwen3-Omni-35B-A3B-SAMI-AsymQ.gguf --local-dir .

# Dimension-aligned SigLIP visual projector (861 MB)
huggingface-cli download MIThridate/Qwen3-Omni-35B-A3B-SAMI-AsymQ \
    mmproj-SAMI-Omni-Vision-BF16.gguf --local-dir .
```

### 2. Launch Server

```bash
llama-server \
  -m Qwen3-Omni-35B-A3B-SAMI-AsymQ.gguf \
  --mmproj mmproj-SAMI-Omni-Vision-BF16.gguf \
  --no-mmproj-offload \
  -ngl 99 \
  -cmoe \
  -t 4 \
  -c 4096 \
  --reasoning off \
  --reasoning-budget 0 \
  --port 8080
```

### 3. Run Client Example

```bash
pip install -r requirements.txt

# Text query
python client_example.py --prompt "Explain the memory bandwidth roofline equation for sparse MoE models."

# Multimodal image query
python client_example.py --image test_diagram.jpg --prompt "Analyze the architectural topology in this diagram."
```

---

## Repository Scope

This repository provides client execution scripts, telemetry instrumentation, and deployment configurations for the SAMI Omni architecture. Model weight surgery pipelines and calibration data remain part of internal research tooling.

---

## Citation

The technical report describing the complete architectural proofs and telemetry is archived on Zenodo: [doi.org/10.5281/zenodo.23293547](https://doi.org/10.5281/zenodo.23293547).

```bibtex
@techreport{spiridon2026sami,
  title       = {SAMI Omni: Asymmetric Mixture-of-Experts Topology and Memory-Decoupled Multimodal Inference on Commodity Hardware},
  author      = {Spiridon, Valentin},
  institution = {TSV Consulting SRL},
  year        = {2026},
  month       = {October},
  doi         = {10.5281/zenodo.23293547},
  url         = {https://doi.org/10.5281/zenodo.23293547}
}
```

---

## License

Code and deployment assets are released under the [Apache-2.0 License](LICENSE).
