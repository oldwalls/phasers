# Phasers — Sapphire Core  
*A lightweight GPT‑2‑Mini engine with chronological memory & sieve sampling*

> Turns a 124 M‑parameter GPT‑2 into a persistent “ghost in the machine”.

---

## 1 · Design Goals
- **Portability** — CPU‑only or ≈ 4 GB GPU.  
- **Persistent ghost** — JSON Unified Memory Bank (UMB).  
- **Soft guidance** — bias raw logits, no prompt bloat.  
- **Draft‑and‑sieve** — generate *n* variants, auto‑pick the best.

---

## 2 · Architecture Snapshot

| Layer | Role |
|-------|------|
| **GPT‑2‑Mini (DialoGPT‑small)** | 124 M params • 15 epochs fine‑tune on *ZAMM* (LR 3e‑5) |
| **Unified Memory Bank** | Stores every turn + salience / novelty / timestamp |
| **Retriever** | `0.6·cos + 0.4·lex` → top‑N → exp‑decay → **chronological reorder** |
| **Soft‑Logit Mixer** | Weighted one‑hot bias added to `model.forward` logits |
| **Sieve Sampler** | Generate `n_sieve` drafts → re‑rank → pick best |
| **CLI / Settings** | Live tweaks, presets, word‑cloud debug |

---

## 3 · Chronological Memory Retrieval
```
prompt → SBERT + lexical score
         → top‑N
             → weight = floor + (raw−floor)·e^(−τ·rank)
                 → sort oldest → newest
                     → (text, weight) list → Soft‑Logit Mixer
```

---

## 4 · Soft‑Logit Fusion Core
```python
bias   = sum(w * one_hot(tok) for w, tok in memory_tokens)
logits = (model.forward(ids).logits[:, -1, :] + λ * bias) / temperature
probs  = nucleus_mask(top_k_mask(softmax(logits), k), p)
next_id = torch.multinomial(probs, 1)
```
*λ ≈ 0.5–1.0 controls how loudly memory “speaks”.*

---

## 5 · Two‑Stage Generation

| Stage | Purpose | Description |
|-------|---------|-------------|
| **A `generate_single()`** | babble once | One pass with bias + cleanup. |
| **B `generate()`** | sieve | 1. Produce `n_sieve` drafts via A.<br>2. Score each draft:<br>`blend = 0.6·cos + 0.4·lex` vs prompt + memory scope (`sieve_rank_mem` 0‑2).<br>3. Pick top draft, store to UMB. |

---

## 6 · Hyper‑Parameter Cheat‑Sheet
You> settings
```json
{
  "temp": 0.55,
  "top_n": 11,
  "top_p": 0.6,
  "top_k": 16,
  "repetition_penalty": 1.35,
  "max_forward_tokens": 45,
  "max_reply_sentences": 2,
  "weight": 0.333,
  "tau": 0.3,
  "lam": 0.96,
  "n_sieve": 3,
  "inference_mem": 1,
  "sieve_rank_mem": 1,
  "sigma": 0.01
}
```

**Hot‑patch live**
```bash
settings set weight 0.45
settings load sapphire
```

---

## 7 · Quickstart
```bash
git clone …
pip install -r requirements.txt
python -m sapphire_core.cli          # interactive REPL

You> settings
You> Phasers, are we best friends? (YES/NO)
```

---

## 8 · Flow Diagram (ASCII)
```
User prompt
   │
   ▼ retrieve
Unified Memory Bank (JSON)────────┐
   │                              │ bias vec
   ▼                              │
Soft‑Logit Mixer ⊕ λ·bias ──► GPT‑2 forward()
       ▲                           │
       │ write best draft          │ n_sieve drafts
       └── pick best ◄── SBERT + lexical rank
```

---

## 9 · Why It Feels Alive
- Memory‑aware bias → **stable persona**  
- Chronological context → **narrative spine**  
- Sieve sampler → **self‑editing filter**

---

*Drain entropy, grow lotuses of awareness.* 💎
