# R‑VFiD (Semantic–Frequency–Prompt) Implementation Guide

> **Goal** — Re‑implement the full R‑VFiD pipeline in Python/PyTorch so it can be trained, evaluated and extended for AAAI 2026 submission.

---

## ✅ Progress Checklist

> Tick every box when the sub‑task is done.  
> **Bold** = mandatory for MVP, 🔄 = iterative refinement.

### 0 Environment & dependencies

### 1 Data loader & augmentation

### 2 Zero‑shot CLIP top‑_c_ classifier

### 3 Router‑Prompt pool builder

### 4 Text encoder forward & prompt fine‑tuning

### 5 Vision backbone (frozen ViT‑B/16)

### 6 LoRA frequency‑expert injection

### 7 Cross‑attention router α‑weights

### 8 Fusion (concat + SE) & classifier head

### 9 Losses

### 10 Training loop & scheduler

### 11 Inference single‑pass ensemble

### 12 Continual‑learning update

### 13 Logging, metrics & visualisation

### 14 Checkpointing & model‑card

---

## 1 System Overview

R‑VFiD fuses **high‑level semantics, low‑level frequency cues and prompt‑based PEFT**. The computation graph (batch size _B_, feature dim _d_ = 768):

```
 x ∈ B × 3 × 224 × 224 ─▶ E_v (ViT‑B/16, frozen) ─▶ V_tok (B × T × d)
                                      │
                                      ├─► K LoRA experts {F_i ∈ B × d}
                                      │
 Router‑Prompt pool P_r (B × L' × d) ──┤
                                      ▼
                         Cross‑Attention Router ─▶ α (B × K)
                                      ▼
                V_freq = Σ α_i F_i  (B × d)
                                      ▼
     V_fuse = SE([V_cls ∥ V_freq])  (B × 2d)
                                      ▼
                LoRA Head → logits (B × 2)
```

Parameter economy: Δθ ≈ **2–3 %** of CLIP.

---

## 2 Module‑by‑Module Specification

|Stage|Purpose|Input shape|Output shape|Trainable params|
|---|---|---|---|---|
|0 Pre‑proc|RGB → CLIP normalised tensor|B×3×H×W|B×3×224×224|—|
|1 Zero‑shot top‑_c_ labels|Image‑conditioned semantics|B×3×224×224|list(len = c)|—|
|2 Router‑Prompt pool|Build _L' = 2c·7_ tokens|list|B×L'×d|✔ (token emb)|
|3 Text Encoder E_t|map tokens→embs|B×L'×d|B×L'×d|frozen|
|4 Vision backbone E_v|semantic tokens|B×3×224×224|V_tok (B×T×d)|frozen|
|5 LoRA expert _i_|low‑level cue _Ci_|V_tok|F_i (B×d)|✔ (r = 4, α = 8)|
|6 Router X‑Attn|α‑weights|P_r, V_tok|α (B×K)|✔|
|7 Freq sum|cue aggregation|α,{F_i}|V_freq (B×d)|—|
|8 Concat + SE|fuse hi/lo|V_cls,V_freq|V_fuse (B×2d)|✔(tiny)|
|9 LoRA Head|binary logits|V_fuse|B×2|✔|

### 2.1 Router Prompt (read‑only)

- **Length L = 7** tokens per sentence, duplicated for _2c_ real/fake templates. 
    
- Updated **only** at embedding level; CLIP weights untouched.
    

### 2.2 LoRA Frequency Experts

- Injected into **Q,K,V** of every ViT block:  
    `W_qkv ← W_qkv + α/r · B A` (rank _r_ = 4).
    
- Three recommended cues: **NPR**, **DnCNN residual**, **NoisePrint**.
    

### 2.3 Cross‑Attention Router

- `Q = P_r W_q , K = V_tok W_k , V = V_tok W_v` → `α = softmax(QKᵀ/√d)`.
    
- Single‑head sufficient; output size _(B,K)_.
    

### 2.4 Fusion & Head

- `V_cat = [V_cls ∥ V_freq]` (_B×2d_).
    
- Squeeze‑and‑Excite: `s = σ(W₂ δ(W₁ Avg(V_cat))) ; V_fuse = s ⊙ V_cat`.
    
- Head: LoRA‑adapted linear → logits.
    

---

## 3 Loss Functions

|   |   |   |
|---|---|---|
|Loss|Formula|Weight|
|Binary CE|−[y log p + (1−y) log (1−p)]|1.0|
|InfoNCE|sim(P_r, V_sem) + sim(P_r, V_freq)|0.1|
|Gating Entropy|−Σ α log α|0.01|

Total: `L = L_CE + 0.1 L_NCE + 0.01 L_ent`.

---

## 4 Training Pipeline

1. **Load** frozen CLIP (ViT‑B/16) weights.
    
2. Build **Dataset & DataLoader** with on‑the‑fly low‑level extraction.
    
3. **Forward pass** per Sec. 2; accumulate losses.
    
4. **Optimizer**: AdamW, lr 1e‑3, weight‑decay 0.01, cosine anneal.
    
5. **Epochs**: 20 per generator domain.
    
6. **Validation** on AIGCDetect; log Acc, AUC, AP.
    

---

## 5 Inference (single‑pass ensemble)

```
with torch.no_grad():
    logits = model(x, prompt_pool_all_domains)  # shape B×2
    p_fake = torch.sigmoid(logits[:,1])
```

- If domain‑prob classifier (k‑means on CLS) is desired, scale task scores before argmax.
    

---

## 6 Continual‑Learning Update

1. **Freeze** all existing prompts & experts.
    
2. **Add** new read‑only prompt _p_k+1_ and (optionally) new LoRA expert.
    
3. **Finetune** only these additions on new generator data (≤5 epochs).
    
4. **Deploy**: still one forward pass, as prompts are independent. 
    

---

## 7 Key Hyper‑parameters

|   |   |   |
|---|---|---|
|Name|Value|Note|
|Image size|224²|CLIP default|
|Patch T|197|1 CLS + 14×14|
|Prompt top‑c|5|zero‑shot classes|
|Prompt len L|7|tokens per sentence|
|LoRA rank r|4|experts & head|
|LoRA α|8|scaling factor|
|K experts|3|NPR, DnCNN, NoisePrint|
|Batch size|64|Fit 24 GB GPU|
|Base lr|1e‑3|AdamW|

---

## 8 Reference Implementations & Assets

- Prompt2Guard (official PyTorch) ▶ GitHub `laitifranz/Prompt2Guard` 
    
- ALEI (Adaptive Low‑level Experts Injection) ▶ GitHub link TBD 
    

Use these repos for data loaders and low‑level cue extraction utilities.

---

## 9 Appendix A — Complete Tensor I/O Flow

|   |   |   |
|---|---|---|
|Stage|Tensor symbol|Shape (B= batch, d=768)|
|Pre‑proc|x|B × 3 × 224 × 224|
|E_v ↓|V_tok|B × T × d|
|LoRA_i ↓|F_i|B × d|
|P_r|—|B × L' × d|
|Router|α|B × K|
|Sum|V_freq|B × d|
|Concat|V_cat|B × 2d|
|SE|V_fuse|B × 2d|
|Head|logits|B × 2|

> **Tip** — Run `torchsummary` or `fvcore.nn.FlopCountAnalysis` to verify parameter & FLOPs budgets.

---

Happy coding & good luck with the AAAI 2026 deadline! 🎯