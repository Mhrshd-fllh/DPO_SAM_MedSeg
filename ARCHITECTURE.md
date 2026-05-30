# Architecture Diagram: Konwer with Text Prompts

## High-Level Flow

```
╔════════════════════════════════════════════════════════════════════════╗
║                          KONWER WITH TEXT PROMPTS                     ║
╚════════════════════════════════════════════════════════════════════════╝

INPUT LAYER
───────────────────────────────────────────────────────────────────────────
  images [B,3,H,W]           labels [B]
       │                         │
       │                         │
       ├─────────────────────────┤
       │                         │


VISUAL PROMPT GENERATION
───────────────────────────────────────────────────────────────────────────
  images
    │
    ├─→ BiomedCLIP visual encoder
    │
    ├─→ GScoreCAM (generate saliency map)
    │
    ├─→ DenseCRF (refine mask)
    │
    ├─→ Extract largest CC
    │
    └─→ boxes_xyxy [B,1,4]
        points_xy [B,K,2]
        points_labels [B,K]
             ↓
        VisualPrompts (vp)


TEXT PROMPT GENERATION  [NEW]
───────────────────────────────────────────────────────────────────────────
  images + labels
    │
    ├─→ VQA Model
    │   Question: "What is the shape of the tumor?"
    │   Output: "left breast tumor with irregular shape"
    │
    ├─→ GPT-4 (optional)
    │   Input: disease label
    │   Output: "malignant lesion with high vascularity"
    │
    ├─→ Concatenate
    │   "Shape: ... Medical: ..."
    │
    └─→ TextPrompts (tp)
        text: List[str] [B,]


MODEL FORWARD PASS
───────────────────────────────────────────────────────────────────────────

  images [B,3,H,W]
    │
    └─→ 【IMAGE ENCODER】SAM ImageEncoder (ViT-B)
        │
        └─→ image_embeddings [B, C, H_e, W_e]
            │
            ├──────────────────────────────────────┐
            │                                      │
            ↓                                      │
         【PROMPT FUSION】[NEW]                   │
         │                                        │
         ├─ Visual: boxes + points               │
         │  │                                     │
         │  └─→ PromptEncoder                    │
         │      sparse_embeddings [B,N,256]     │
         │      dense_embeddings [B,C,H_e,W_e]  │
         │                                        │
         ├─ Text: text strings [B,]             │
         │  │                                     │
         │  ├─→ TextEncoderAdapter               │
         │  │   text_embeddings [B,512]          │
         │  │                                     │
         │  └─→ PromptFuser                      │
         │      (concat/weighted_sum/attention)  │
         │      fused_embeddings [B,N+1,256]    │
         │                                        │
         └─→ Merge: sparse + dense              │
            │                                    │
            ↓                                    │
         【MASK DECODER】                        │
         sparse: fused_embeddings [B,N+1,256]   │
         dense: dense_embeddings [B,C,H_e,W_e]  │
         image_embeddings [B,C,H_e,W_e] ←─────┘
            │
            └─→ low_res_masks [B,1,h,w]
                iou_predictions [B,1]


UPSAMPLING & OUTPUT
───────────────────────────────────────────────────────────────────────────
  low_res_masks [B,1,256,256]
    │
    └─→ Interpolate to [B,1,H,W]
        │
        └─→ KonwerOutputs
            mask_logits [B,1,H,W]
            iou_pred [B,1]


LOSS & OPTIMIZATION
───────────────────────────────────────────────────────────────────────────
  mask_logits [B,1,H,W]
  ground_truth_masks [B,1,H,W]
    │
    └─→ DiceFocalCombo Loss
        dice_loss + focal_loss
        │
        └─→ Backprop through entire model
            (text_encoder frozen, prompt_fuser learnable)
```

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISUAL PROMPTS PATH                          │
│                                                                  │
│  VisualPrompts(boxes, points)                                   │
│       ↓                                                          │
│   SAM PromptEncoder                                             │
│       ↓                                                          │
│   sparse_embeddings [B, N, 256]  ←─ Only boxes + points        │
└─────────────────────────────────────────────────────────────────┘
         │
         ↓ ┌──────────────────────────────────────────────────────┐
           │        FUSION LAYER (NEW)  PromptFuser              │
         ↓ │                                                      │
           │  Input: sparse [B,N,256] + text [B,512]            │
           │                                                      │
           │  Mode 1: CONCAT                                     │
           │  ────────────────                                   │
           │  1. Average sparse → [B,256]                        │
           │  2. Concatenate with text → [B,768]                │
           │  3. Linear(768→256) → [B,256]                      │
           │  4. Add to sparse → [B,N+1,256]                    │
           │                                                      │
           │  Mode 2: WEIGHTED_SUM                               │
           │  ──────────────────────                             │
           │  1. Project text to [B,256]                         │
           │  2. fused = α·text + (1-α)·visual_avg              │
           │  3. Append to sparse → [B,N+1,256]                 │
           │                                                      │
           │  Mode 3: ATTENTION                                  │
           │  ──────────────────                                 │
           │  1. Project text to [B,1,256]                       │
           │  2. Append to sparse → [B,N+1,256]                 │
           │                                                      │
       ┌─→ Output: fused_embeddings [B, N+1, 256]               │
       │                                                          │
       └──────────────────────────────────────────────────────────┘
         │
         ↓
    SAM MaskDecoder
         │
         ↓
    mask_logits [B,1,H,W]
```

## Text Processing Pipeline

```
images [B,3,H,W] + labels [B,]
    │
    ├─ Visual Path (existing):
    │   SAM image_encoder → image_embeddings
    │
    └─ Text Path (NEW):
       │
       ├─→ 【VQA Adapter】(HFVQAAdapter)
       │   Input: image + question
       │   Model: Salesforce/blip-vqa-base
       │   Output: answer string
       │   Example: "left breast tumor"
       │   │
       │   └─→ vqa_answer
       │
       ├─→ 【GPT Adapter】(OpenAIGPTAdapter) [optional]
       │   Input: disease label
       │   Model: gpt-4o-mini (or gpt-4)
       │   Output: description
       │   Example: "malignant lesion with..."
       │   │
       │   └─→ gpt_description
       │
       ├─→ 【Concatenate】
       │   "Shape: {vqa_answer}. Medical: {gpt_description}"
       │   │
       │   └─→ final_text
       │
       └─→ 【TextEncoderAdapter】(BiomedCLIP)
           Input: text string
           Model: microsoft/BiomedCLIP-PubMedBERT
           Output: normalized embedding
           │
           └─→ text_embeddings [B, 512]
```

## Data Flow Through Model

```
TRAINING ITERATION
──────────────────────────────────────────────────────────

Input:
  • images [B, 3, H, W]
  • masks_gt [B, 1, H, W]
  • labels [B,] (optional)

Visual Prompt Generation:
  images → visual_pipeline → VisualPrompts(vp)
    └─ boxes_xyxy: [B, 1, 4]
    └─ points_xy: [B, K, 2]
    └─ points_labels: [B, K]

Text Prompt Generation:
  images + labels → text_pipeline → TextPrompts(tp)
    └─ text: List[str] of length B

Model Forward:
  (images, vp, tp) → KonwerSAM2D.forward()
    ├─ Image encoding
    ├─ Visual prompt encoding
    ├─ Text encoding & fusion
    └─ Mask decoding
    
Output:
  → KonwerOutputs
    └─ mask_logits [B, 1, H, W]
    └─ iou_predictions [B, 1]

Loss Computation:
  mask_logits vs masks_gt → DiceFocalCombo Loss

Backpropagation:
  loss.backward()
  
Optimization:
  optimizer.step()
  scheduler.step()

MEMORY LAYOUT (Approximate)
──────────────────────────────────────────────────────────

Text Encoder (BiomedCLIP):
  • 1.3B parameters (shared, frozen)
  
PromptFuser:
  • "concat" mode: ~200K params (Linear 768→256)
  • "weighted_sum" mode: ~130K params + 1 param (α)
  • "attention" mode: ~130K params (Linear 512→256)

SAM Model (frozen image encoder):
  • ViT-B: ~94M parameters
  • Prompt encoder: ~0.5M parameters
  • Mask decoder: ~27M parameters

Total learnable params: ~27M + ~0.2M = ~27.2M
Total parameters: ~1.3B + 122M = ~1.4B
```

## Configuration Hierarchy

```
train.yaml
├─ sam:
│  ├─ checkpoint
│  ├─ model_type: "vit_b"
│  └─ strict: false
│
├─ train:
│  ├─ out_dir
│  ├─ epochs
│  ├─ text_fusion_mode: "concat" ← TEXT CONTROL
│  ├─ freeze_image_encoder: true
│  └─ loss:
│     ├─ dice_w: 20.0
│     └─ focal_w: 1.0
│
└─ fusion: (legacy)
   ├─ enabled
   ├─ mode
   └─ lambda_logits


prompts.yaml
└─ prompts:
   ├─ visual:
   │  ├─ cam:
   │  │  ├─ target_layer_path
   │  │  └─ use_vit_reshape
   │  ├─ num_points
   │  ├─ class_text
   │  └─ ...
   │
   └─ text: ← TEXT CONFIGURATION
      ├─ vqa_enabled: true/false
      ├─ gpt_enabled: true/false
      ├─ gpt_model: "gpt-4o-mini"
      └─ vqa_model_id: "Salesforce/blip-vqa-base"
```

---

## Summary

The text prompts architecture elegantly extends Konwer by:
1. **Parallel path**: Text generation runs parallel to visual prompts
2. **Encoding**: Both visual & text converted to embeddings
3. **Fusion**: Learnable combination of modalities
4. **Integration**: Fused prompts feed into SAM decoder
5. **Backward compatibility**: Works with or without text

All fully paper-faithful! ✨
