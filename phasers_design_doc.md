# Phasers — Sapphire Core  
*A lightweight GPT-2-Mini engine with chronological memory & sieve sampling*

> This document explains **how a 124 M-parameter GPT-2 model turns into a persistent “ghost in the machine”.**

---

## 1 · Design Goals
* **Portability** — runs CPU-only or on ≈ 4 GB GPU.  
* **Persistent ghost** — grows a JSON Unified Memory Bank (UMB).  
* **Soft guidance** — bias raw logits rather than stuffing huge prompts.  
* **Draft-and-sieve** — generate *n* variants, auto-select the best.

---

## 2 · Architecture Snapshot

| Layer | Role |
|-------|------|
| **GPT-2-Mini (DialoGPT-small)** | 124 M params – fine-tuned 15 epochs on *Zen and the Art* (LR 3 e-5) |
| **Unified Memory Bank** | Stores every turn + salience, novelty, timestamp |
| **Retriever** | `0.6 · cos + 0.4 · lex` → top-N → exp-decay → **chronological reorder** (old→new) |
| **Soft-Logit Mixer** | Adds weighted one-hot bias to `model.forward` logits |
| **Sieve Sampler** | Generate `n_sieve` drafts → re-rank → pick best |
| **CLI / Settings** | Live tweak, presets, word-cloud debug |

---

## 3 · Chronological Memory Retrieval
```text
prompt → SBERT + lexical score
         → top-N
             → weight = floor + (raw−floor) · e^(−τ·rank)
                 → sort oldest → newest
                     → (text, weight) list → Soft-Logit Mixer
