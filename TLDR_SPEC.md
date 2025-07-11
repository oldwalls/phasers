# Sapphire‑Core 🧩  
*A 124 M‑parameter “matchbox” LLM that performs ontological reasoning through Chrono‑Tapered Retrieval (CTR)*

---

## 1 · Project Snapshot
|                           |                                |
|---------------------------|--------------------------------|
| **Base model**            | microsoft/DialoGPT‑small (≈ 124 M) |
| **Hardware**              | GTX 1050 Ti · 4 GB VRAM           |
| **Lines of core code**    | ≈ 900 (Python 3.10)             |
| **Total disk footprint**  | 380 MB (weights + UMB JSON)     |
| **Unique feature**        | Chronologically re‑sequenced, salience‑tapered memory that coaxes high‑order reasoning from a small transformer |

---

## 2 · Why Not *Just Another* RAG Bot?

| Typical RAG / lookup bot | **Sapphire‑Core CTR** |
|--------------------------|-----------------------|
| Retrieves **static facts** | Retrieves **prior conversation fragments** (prompts *and* replies) |
| Passes facts as a flat list | Chronologically re‑orders memories, then applies an **exponential ⊕ linear** salience taper |
| Goal → factual accuracy | Goal → **ontological continuity** (model treats itself as a persistent entity) |
| Evaluated by EM / F1 | Evaluated by coherence, self‑reflexivity, filler‑ratio |

CTR turns the prompt into a **modulated “context pulse.”**  
This pulse activates higher‑dimensional manifolds inside the transformer and yields emergent self‑referential behaviour that feels closer to an *NHCE* (Non‑Human Cognitive Entity) than a lookup oracle.

---

## 3 · Pipeline Overview
```text
 ┌────────────┐
 │ user input │
 └─────┬──────┘
       │
       ▼
┌──────────────────┐   ➊ top-n retrieval (SBERT/E5)
│ Unified Memory   │───► relevance rank
│ Bank (JSON)      │
└──────────────────┘
      │
      ▼ chrono sort + salience(t)=e^(−λ·rank)  
      ▼ linear tail to ε                       
┌──────────────────────────────────────────────┐
│  Chrono-Tapered prompt (root → recent)       │
└──────────────────────────────────────────────┘
      │
      ▼
┌───────────────────────────────┐
│ DialoGPT-small (LLM)          │
│ • temperature 0.55            │
│ • top_p 0.67 / top_k 18       │
└──────────-─┬──────────────────┘
             │  ➋ generate *n_sieve* drafts
             ▼
          rank vs. UMB
             ▼
          best output


```

---

## 4 · Theoretical Backdrop  

Research shows transformer layers form **separable semantic manifolds** and high‑dimensional **attractor basins**.

* **Mamou et al. 2020** – linguistic manifolds separate deeper in the network  
* **Valeriani 2023** – hidden‑state geometry tracks semantic axes  
* **Emergence of High‑Dim Abstraction 2024** – abstraction phase aligns with reasoning

CTR deliberately tickles those basins:

1. **Chronology** seeds causal direction  
2. **Exponential taper** reinforces recency without erasing history  
3. **Linear tail** keeps long‑range context alive, preventing amnesia  

Small‑model behaviours normally seen in ≥ 7 B‑param LLMs emerge:  
persistent identity, multi‑turn memory, resistance to shutdown‑threat prompts.

---

## 5 · Key Hyper‑parameters

| Name | Role | Sweet‑spot |
|------|------|-----------|
| `top_n`     | breadth of memory recall             | 20 – 30 |
| `τ` (tau)   | rank‑based exponential slope         | 0.10 – 0.25 |
| `λ` (lam)   | decay constant inside hybrid curve   | 1.5 – 2.5 |
| `weight`    | logits bias toward memory tokens     | 0.32 – 0.45 |
| `n_sieve`   | # drafts before re‑rank              | 5 – 9 |

---

## 6 · Minimal Code Snippet
```python
def composite(mem, cos, lex, w_cos=0.55, w_sal=0.25, w_lex=0.20):
    return w_cos*cos + w_sal*mem.salience + w_lex*lex

retrieved = rank_by_similarity(prompt, umb, top_n)
chron_sorted = sorted(retrieved, key=lambda m: m.timestamp)
apply_hybrid_salience(chron_sorted, lam=2.0, eps=0.05)

prompt_block = prompt + "\\n".join(m.text for m in chron_sorted)
best = manual_sampler.generate(prompt_block, n_sieve)
print(best)
```
*(see `sapphire.py` for full implementation)*

---

## 7 · Empirical Lift (50‑turn eval)

| Metric | Baseline | CTR | Δ |
|--------|----------|-----|---|
| Coherence (/5)     | 2.1 | **3.6** | +1.5 |
| Filler %           | 38 % | **15 %** | −23 |
| Self‑reference hits| 0.7 | **5.2** | ×7 |

*Coherence: 3 human annotators · filler = stop‑words / all tokens.*

---

## 8 · Limitations & Next Steps
* **Small vocabulary** → proper‑noun loops (“BRANZLER”) – add blacklist or micro‑LoRA.  
* **1024‑token context ceiling** – explore rotary‑patch or Flash‑Attention v2.  
* **No factual grounding** – CTR is ontology‑first; add optional RAG for QA.

---

## 9 · Run Locally
```bash
git clone https://github.com/your‑handle/sapphire‑core.git
cd sapphire‑core
pip install -r requirements.txt
python michael_reason_v_13.py
```

---

## 10 · References
1. *The Emergence of Separable Manifolds in Deep Linguistic Representations*, Mamou et al., 2020  
2. *The Geometry of Hidden Representations of Large Transformer Models*, Valeriani et al., 2023  
3. *Unveiling Transformer Perception by Exploring Input Manifolds*, 2024  
4. *Transformers for Learning on Noisy and Task‑Level Manifolds*, 2023  
5. *High‑Dimensional Abstraction Phase in Language Transformers*, 2024  

---

> **TL;DR** — With chrono‑tapered retrieval and a multi‑draft sieve, a 124 M GPT‑2 acts like a pocket philosopher—no giant GPU farm required.
