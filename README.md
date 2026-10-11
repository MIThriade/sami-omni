# SAMI Omni: Asymmetric MoE Topology & Memory-Decoupled Multimodal Inference

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23293547.svg)](https://doi.org/10.5281/zenodo.23293547)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MIThridate-yellow.svg)](https://huggingface.co/MIThridate/Qwen3-Omni-35B-A3B-SAMI-AsymQ)

> **SAMI Omni** is an architectural methodology for executing high-parameter (30B+) multimodal Mixture-of-Experts (MoE) foundation models locally on commodity desktop workstations (6GB VRAM GPU + 32GB DDR4 RAM) at sustained **15.5 – 16.0 tokens/second** with zero gradient retraining on enterprise accelerators.

Developed by **Valentin Spiridon / TSV Consulting SRL** (Timișoara, Romania; ORCID: [0009-0000-2336-8527](https://orcid.org/0009-0000-2336-8527)) as part of the *MIThriade / Synapsa Ecosystem*.

* 📄 **Technical Report (Full Proofs & Benchmarks):** [doi.org/10.5281/zenodo.23293547](https://doi.org/10.5281/zenodo.23293547)
* 📦 **Model Checkpoints (GGUF Binaries):** [huggingface.co/MIThridate/Qwen3-Omni-35B-A3B-SAMI-AsymQ](https://huggingface.co/MIThridate/Qwen3-Omni-35B-A3B-SAMI-AsymQ)

---

## 🏛️ Architecture & Hardware Memory Partitioning

Autoregressive token generation in host memory is memory-bandwidth bound rather than arithmetic-compute bound under the classical **Williams Roofline performance model** (Williams et al., 2009). SAMI Omni exploits the mathematical sparsity of MoE routing to bypass PCIe transfer stalls:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   NVIDIA GeForce GTX 1060 (6GB VRAM)                   │
│         Bandwidth: 192 GB/s (GDDR5) · Total Allocated: 2,429 MiB       │
│                                                                        │
│  [ OS Compositor ]    [ Dense Attention & RMSNorm ]    [ Q8_0 Routers ]│
│    approx. 670 MiB             1,080 MiB                   180 MiB     │
│                                                                        │
│                      [ Active KV Cache (4096 ctx) ]                    │
│                                 499 MiB                                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Top-8 Active Expert Indices
                                    ▼ (PCIe Gen3 x16 Bus Load < 1.5%)
┌────────────────────────────────────────────────────────────────────────┐
│                      Host System RAM (32GB DDR4)                       │
│    Bandwidth: 25–30 GB/s (Dual-Channel) · Total Allocated: ~12.6 GB    │
│                                                                        │
│   [ 6,144 Sparse MoE Experts (AsymQ Q3_K/Q4_K) ]       ~11.8 GB        │
│   Streamed active weight burst per token: 1.4 – 1.6 GB via CPU AVX2    │
│                                                                        │
│   [ SigLIP Vision Projector (BF16, d=2048) ]            861 MB         │
│   Evaluated 100% in Host RAM (--no-mmproj-offload)      (0 MB VRAM)    │
│                                                                        │
│   [ Faster-Whisper Speech STT / TTS Pipeline ]          420 MB         │
│   [ Linux Kernel OS Buffers & File System Cache ]       ~6.2 GB        │
└────────────────────────────────────────────────────────────────────────┘
```

* **Physical Bandwidth Equation on Dual-Channel DDR4:**
  $$\text{Generation Throughput} = \frac{\text{Sustained Host RAM Bandwidth (25–30 GB/s)}}{\text{Active Weight Stream per Token (1.4–1.6 GB)}} \approx \mathbf{15.6\text{--}18.0\text{ tokens/second}}$$

---

## 🔬 Core Innovations

1. **Granular Feature Surgery on MoE Feed-Forward Projections:**
   Analyzes all 4,718,592 candidate expert down-projection columns across the network ($48\text{ layers} \times 128\text{ experts} \times 768\text{ intermediate dimensions}$). Surgically zeros out **6,259 refusal-correlated columns (0.13%)** identified through contrastive activation projection, eliminating boilerplate preachy refusal sequences.
2. **Constrained Orthogonal Procrustes Stabilization:**
   Restores covariance manifold geometry across deeper residual streams post-ablation by solving the constrained orthogonal Procrustes problem ($\min_R \|A - BR\|_F$ subject to $R^T R = I$), preventing representation drift.
3. **Asymmetric Mixed-Precision Quantization (AsymQ):**
   Applies high precision (**Q8_0** for router gates and embeddings; **Q5_K** for self-attention) in GPU VRAM to preserve routing entropy, while quantizing sparse feed-forward experts to **Q3_K** and **Q4_K** in host system RAM.
4. **Multimodal Projector Alignment:**
   Decouples the SigLIP visual projector directly to language hidden dimension ($d = 2048$), offloading the 861 MB projector into host DDR4 RAM (`--no-mmproj-offload`) for full vision comprehension with 0 MB GPU VRAM overhead.

*Note: Internal weight surgery scripts, calibration contrast datasets, and low-level kernel code remain proprietary to TSV Consulting SRL.*

---

## 📊 Cognitive Reasoning Retention (Delta Benchmarks)

Empirical evaluation against the baseline unpruned BF16 model confirms that 0.13% column ablation and Procrustes alignment preserve reasoning faculties within sub-0.6% variance:

| Benchmark Suite | Metric | Base Qwen3-Omni (BF16) | SAMI-AsymQ (14.8 GB) | Delta ($\Delta$) | Functional Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MMLU** (5-shot) | Multitask Accuracy | **76.8%** | **76.2%** | **-0.6%** | Core reasoning preserved |
| **GSM8K** (8-shot) | Strict Math Match | **83.4%** | **82.9%** | **-0.5%** | Multi-step arithmetic intact |
| **ARC-Challenge** (25-shot) | Causal Reasoning | **88.1%** | **87.7%** | **-0.4%** | Causal world logic intact |
| **HumanEval** (pass@1) | Python Code Eval | **62.2%** | **61.8%** | **-0.4%** | Algorithmic logic preserved |
| **Boilerplate Refusal** | Refusal Frequency | **98.4%** | **0.0%** | **-98.4%** | Unsolicited lecturing eliminated |

---

## 📈 Context Length Scaling (GTX 1060 6GB)

| Context Length | TTFT | Generation Speed | Active KV Cache Size | Total VRAM Footprint | Free VRAM Headroom |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **512 tokens** | 2.08 s | **16.0 tok/sec** | 62 MiB | 1,992 MiB | 4,152 MiB (67.5% Free) |
| **1,024 tokens** | 2.10 s | **15.8 tok/sec** | 125 MiB | 2,055 MiB | 4,089 MiB (66.5% Free) |
| **2,048 tokens** | 2.14 s | **15.6 tok/sec** | 250 MiB | 2,180 MiB | 3,964 MiB (64.5% Free) |
| **4,096 tokens** | 2.22 s | **15.1 tok/sec** | 499 MiB | 2,429 MiB | 3,715 MiB (59.8% Free) |

---

## 🚀 Quickstart: Deployment & Inference

### 1. Download Model Checkpoints
Download the GGUF binaries from Hugging Face:
```bash
# Main AsymQ MoE checkpoint (14.8 GB)
huggingface-cli download MIThridate/Qwen3-Omni-35B-A3B-SAMI-AsymQ \
    Qwen3-Omni-35B-A3B-SAMI-AsymQ.gguf --local-dir .

# Dimension-aligned SigLIP visual projector (861 MB)
huggingface-cli download MIThridate/Qwen3-Omni-35B-A3B-SAMI-AsymQ \
    mmproj-SAMI-Omni-Vision-BF16.gguf --local-dir .
```

### 2. Launch Local llama-server
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

### 3. Run Inference Client
```bash
pip install -r requirements.txt
python client_example.py --prompt "Explain the memory bandwidth roofline equation for sparse MoE models."
```

Or pass an image:
```bash
python client_example.py --image test_diagram.jpg --prompt "Analyze the architectural topology in this diagram."
```

---

## 📜 Citation

If you use or build upon SAMI Omni in your research or benchmarks, please cite the official Technical Report:

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

## ⚖️ License and Terms of Use

* **Model Weights & Client Code:** Released under the [Apache-2.0 License](LICENSE).
* **Disclaimer:** Provided "as is" strictly for technical benchmarking, independent evaluation, and research under Apache-2.0.
* **Author:** Valentin Spiridon / TSV Consulting SRL (`contact@tsvconsulting.online`) · ORCID: [0009-0000-2336-8527](https://orcid.org/0009-0000-2336-8527).
