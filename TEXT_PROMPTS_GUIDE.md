# Text Prompts Integration Guide (Konwer Paper-Faithful)

## Overview

This implementation adds **text prompt support** to Konwer following the paper's methodology:

> **Text prompts:** MedVInT-style VQA answer + GPT-4 generic description → concatenated text prompt

## Architecture

### Components

1. **TextEncoderAdapter** (`prompts/text/text_encoder.py`)
   - Encodes text strings to BiomedCLIP embeddings
   - Input: `List[str]` of length B
   - Output: `[B, 512]` embeddings

2. **PromptFuser** (`models/prompt_fuser.py`)
   - Fuses text embeddings with visual prompt embeddings
   - Supports 3 fusion modes:
     - **concat**: Concatenate and project back to prompt dimension
     - **weighted_sum**: Learnable weighted combination
     - **attention**: Simple attention-based fusion

3. **KonwerSAM2D** (updated `models/konwer_sam2d.py`)
   - Now accepts optional `TextPrompts` (tp parameter)
   - Internally uses TextEncoderAdapter + PromptFuser
   - Backward compatible (works without text prompts)

## Usage

### Training with Text Prompts

```python
# Initialize text encoder
from prompts.visual.load_biomedclip import load_biomedclip
from prompts.text.text_encoder import TextEncoderAdapter

clip_model, _, tokenizer = load_biomedclip(device="cuda")
text_encoder = TextEncoderAdapter(model=clip_model, tokenizer=tokenizer, device="cuda")

# Initialize model with text support
model = KonwerSAM2D(
    sam,
    text_encoder=text_encoder,
    fusion_mode="concat"  # or "weighted_sum", "attention"
)

# Generate text prompts during training
from prompts.text.text_prompt_pipeline import TextPromptPipeline, TextPromptConfig

text_cfg = TextPromptConfig(
    vqa_enabled=True,
    gpt_enabled=True,  # Requires OpenAI API key
    gpt_model="gpt-4o-mini",
)
text_pipeline = TextPromptPipeline(text_cfg, device="cuda")

# Forward pass
tp = text_pipeline(images, labels=labels)  # Generate text prompts
out = model(images, vp=vp, tp=tp)  # Use both visual and text prompts
```

### Training Script

The updated training script (`train/stage1_train.py`) now:

1. ✅ Loads BiomedCLIP for text encoding
2. ✅ Initializes TextEncoderAdapter
3. ✅ Initializes KonwerSAM2D with text support
4. ✅ Generates text prompts from VQA + GPT-4
5. ✅ Fuses text and visual prompts during forward pass

Run training with:
```bash
python train/stage1_train.py \
  --config configs/default.yaml \
  --prompts configs/prompts.yaml \
  --datasets configs/datasets.yaml \
  --train_cfg configs/train.yaml
```

## Configuration

### train.yaml

```yaml
train:
  # Text-visual prompt fusion mode
  text_fusion_mode: "concat"  # "concat", "weighted_sum", or "attention"
```

### prompts.yaml

Configure text prompt generation:
```yaml
prompts:
  text:
    vqa_enabled: true          # Use VQA model
    gpt_enabled: false         # Use OpenAI API
    gpt_model: "gpt-4o-mini"   # GPT model (requires API key in env var OPENAI_API_KEY)
```

## Fusion Modes Explained

### 1. **concat** (Default)
- Concatenates text embedding [B, 512] with averaged visual prompts [B, 256]
- Projects back to prompt dimension via linear layer
- **Best for:** Strong text-visual interaction

### 2. **weighted_sum**
- Learns a weight α to combine text and visual contributions
- `fused = (1-α) * visual_avg + α * text_proj`
- **Best for:** Gradual blending of modalities

### 3. **attention**
- Treats text as a separate prompt token
- Added as [B, 1, D] to visual prompts [B, N, D]
- **Best for:** Modular handling of text information

## Examples

### Example 1: Basic Usage (VQA only, no GPT)

```python
text_cfg = TextPromptConfig(
    vqa_enabled=True,
    gpt_enabled=False,
)
text_pipeline = TextPromptPipeline(text_cfg, device="cuda")

# VQA outputs: "left breast tumor with irregular shape"
tp = text_pipeline(images, labels=None)
out = model(images, vp=vp, tp=tp)
```

### Example 2: Full Paper Setup (VQA + GPT-4)

```python
text_cfg = TextPromptConfig(
    vqa_enabled=True,
    gpt_enabled=True,
    gpt_model="gpt-4o-mini",
)
text_pipeline = TextPromptPipeline(text_cfg, device="cuda")

# Generates:
# "Shape and location: left breast tumor, irregular boundary. 
#  Medical characteristics: malignant with high vascularity"
tp = text_pipeline(images, labels=["malignant"])
out = model(images, vp=vp, tp=tp)
```

### Example 3: Switching Fusion Modes

```python
# Concat mode (strong interaction)
model_concat = KonwerSAM2D(sam, text_encoder=text_encoder, fusion_mode="concat")

# Weighted sum mode (learnable blend)
model_weighted = KonwerSAM2D(sam, text_encoder=text_encoder, fusion_mode="weighted_sum")

# Attention mode (separate tokens)
model_attn = KonwerSAM2D(sam, text_encoder=text_encoder, fusion_mode="attention")
```

## Training Tips

1. **Text prompts improve results** when:
   - Domain labels are available (organ, disease type)
   - VQA model is well-calibrated for your data
   - Text descriptions are meaningful

2. **Disable text prompts** if:
   - Training without labels
   - VQA outputs are noisy/incorrect
   - You want to isolate visual prompt contribution

3. **Fusion mode selection**:
   - **concat**: Better for limited data (more parameters learn interaction)
   - **weighted_sum**: Good balance between modularity and interaction
   - **attention**: Better for very large datasets

## Backward Compatibility

✅ **Model works without text prompts:**
```python
# Old way (still works)
model = KonwerSAM2D(sam)
out = model(images, vp)

# New way (with text)
model = KonwerSAM2D(sam, text_encoder=text_encoder)
out = model(images, vp=vp, tp=tp)
```

## Implementation Details

### TextPrompts Dataclass
```python
@dataclass(frozen=True)
class TextPrompts:
    text: List[str]  # List of text descriptions, length B
```

### Forward Flow
```
images [B,3,H,W]
  ↓
image_encoder → image_embeddings
  ↓
prompt_encoder (boxes + points) → sparse_embeddings [B,N,D]
  ↓
TextEncoderAdapter(text) → text_embeddings [B,512]
  ↓
PromptFuser(sparse_embeddings, text_embeddings) → fused_embeddings [B,N+1,D]
  ↓
mask_decoder → mask_logits [B,1,H,W]
```

## File Structure

```
prompts/
  text/
    text_encoder.py              ← NEW: TextEncoderAdapter
    text_prompt_pipeline.py      ← EXISTING: Generates TextPrompts
    vqa_medvint_adapter.py       ← EXISTING: VQA model
    gpt4_adapter.py              ← EXISTING: GPT-4 wrapper

models/
  konwer_sam2d.py               ← UPDATED: Added text support
  prompt_fuser.py               ← NEW: Fusion module

train/
  stage1_train.py               ← UPDATED: Uses text prompts

configs/
  train.yaml                    ← UPDATED: text_fusion_mode param
```

## Troubleshooting

### Issue: "TextPrompts not imported"
```
ModuleNotFoundError: No module named 'core.types'
```
**Solution:** Make sure you're in repo root and have initialized __init__.py files

### Issue: "text_encoder is None"
```
AttributeError: 'NoneType' object has no attribute 'encode'
```
**Solution:** Pass `text_encoder` when creating KonwerSAM2D:
```python
model = KonwerSAM2D(sam, text_encoder=text_encoder, fusion_mode="concat")
```

### Issue: OOM with text prompts
```
RuntimeError: CUDA out of memory
```
**Solutions:**
1. Reduce batch_size in config
2. Use `fusion_mode="weighted_sum"` (fewer parameters than "concat")
3. Disable text prompts (set `tp=None`)

## Paper Reference

From "Konwer: Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation" (CVPR 2025):

> **Text Prompting Pipeline:**
> 1. VQA: "What is the shape of the tumor?"
> 2. GPT-4: Generic description of disease
> 3. Concatenate: "Shape: ... Medical: ..."

This implementation faithfully reproduces that pipeline.
