# 🚀 Text Prompts Quick Reference Card

## What's New? 

Your Konwer model now supports **text prompts** just like the paper describes!

---

## The 3-Line Overview

```python
# 1. Create text encoder (encodes text → embeddings)
text_encoder = TextEncoderAdapter(clip_model, tokenizer, device="cuda")

# 2. Create model with text support
model = KonwerSAM2D(sam, text_encoder=text_encoder, fusion_mode="concat")

# 3. Generate text prompts and use them
tp = text_pipeline(images, labels=labels)  # VQA + GPT-4
out = model(images, vp=vp, tp=tp)  # Use both visual + text
```

---

## Files You Should Know About

### 📖 Documentation
| File | Purpose |
|------|---------|
| **TEXT_PROMPTS_GUIDE.md** | Complete guide (read this first!) |
| **IMPLEMENTATION_SUMMARY.md** | Overview of what was added |
| **IMPLEMENTATION_CHECKLIST.md** | Detailed verification checklist |

### 📝 Examples  
| File | What It Shows |
|------|---------------|
| **examples_text_prompts.py** | 6 runnable examples |

### 💻 Code
| File | What Changed |
|------|-------------|
| `models/konwer_sam2d.py` | ✎ Added text support to forward() |
| `models/prompt_fuser.py` | ✚ NEW: Fuses text + visual prompts |
| `prompts/text/text_encoder.py` | ✚ NEW: Encodes text to embeddings |
| `train/stage1_train.py` | ✎ Generates & uses text prompts |
| `configs/train.yaml` | ✎ Added text_fusion_mode config |

---

## How It Works (Paper-Faithful)

### Visual Prompts (Already Existed)
```
Image → BiomedCLIP → gScoreCAM → DenseCRF → boxes + points
```

### Text Prompts (Now Added!) ✨
```
Image + Label → VQA: "What is the shape?" → "left breast tumor with irregular..."
            ↓
         GPT-4: "Describe disease" → "malignant lesion with..."
            ↓
         Concatenate: "Shape: ... Medical: ..."
            ↓
         BiomedCLIP encoder → [B, 512] embeddings
```

### Fusion (New!) 🔄
```
Visual [B, N, 256] + Text [B, 512]
         ↓
    PromptFuser (3 strategies)
         ↓
    Fused [B, N+1, 256] → Mask Decoder
```

---

## Configuration

### Enable Text Prompts

**configs/train.yaml:**
```yaml
train:
  text_fusion_mode: "concat"  # or "weighted_sum", "attention"
```

**configs/prompts.yaml** (in prompts.text section):
```yaml
prompts:
  text:
    vqa_enabled: true
    gpt_enabled: true       # Requires OPENAI_API_KEY env var
    gpt_model: "gpt-4o-mini"
```

### Set API Key (for GPT-4)
```bash
export OPENAI_API_KEY="your-api-key"
```

---

## Training Command

```bash
python train/stage1_train.py \
  --config configs/default.yaml \
  --prompts configs/prompts.yaml \
  --datasets configs/datasets.yaml \
  --train_cfg configs/train.yaml
```

---

## Fusion Modes at a Glance

| Mode | Mechanism | When to Use |
|------|-----------|------------|
| **concat** (default) | Text + visual → Linear | Strong interaction needed |
| **weighted_sum** | α·text + (1-α)·visual | Balanced blending |
| **attention** | Text as separate token | Modular design |

---

## API Reference

### TextEncoderAdapter
```python
from prompts.text.text_encoder import TextEncoderAdapter

encoder = TextEncoderAdapter(clip_model, tokenizer, device="cuda")
embeddings = encoder.encode(["tumor", "lesion"])  # [2, 512]
```

### PromptFuser
```python
from models.prompt_fuser import PromptFuser

fuser = PromptFuser(text_dim=512, prompt_dim=256, fusion_mode="concat")
fused = fuser(visual_embeddings, text_embeddings)  # [B, N+1, 256]
```

### KonwerSAM2D (Updated)
```python
from models.konwer_sam2d import KonwerSAM2D

# With text support
model = KonwerSAM2D(sam, text_encoder=encoder, fusion_mode="concat")
out = model(images, vp=vp, tp=tp)

# Without text (still works)
model = KonwerSAM2D(sam)
out = model(images, vp)
```

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'prompts.text.text_encoder'` | Make sure you're in repo root |
| `AttributeError: 'NoneType' object has no attribute 'encode'` | Pass `text_encoder` to KonwerSAM2D |
| `RuntimeError: CUDA out of memory` | Reduce batch_size or use `fusion_mode="weighted_sum"` |
| GPT-4 not responding | Check `OPENAI_API_KEY` environment variable |

---

## What's Backward Compatible? ✅

Your code still works:
```python
# Old way (no text)
model = KonwerSAM2D(sam)
out = model(images, vp)  # ✓ Still works!

# New way (with text)
model = KonwerSAM2D(sam, text_encoder=text_encoder)
out = model(images, vp=vp, tp=tp)  # ✓ Also works!
```

---

## For More Details...

- **Want to understand architecture?** → Read **TEXT_PROMPTS_GUIDE.md**
- **Want to see code examples?** → Run **examples_text_prompts.py**
- **Want complete details?** → See **IMPLEMENTATION_SUMMARY.md**
- **Want to verify nothing broke?** → Check **IMPLEMENTATION_CHECKLIST.md**

---

## Key Takeaway 🎯

Your Konwer model now:
- ✅ Generates text prompts (VQA + GPT-4)
- ✅ Encodes text to embeddings (BiomedCLIP)
- ✅ Fuses with visual prompts (3 strategies)
- ✅ Uses both in training
- ✅ Still works without text (backward compatible)
- ✅ Fully follows the paper

**Ready to train!** 🚀
