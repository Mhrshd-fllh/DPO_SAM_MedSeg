#!/usr/bin/env python
"""
Quick-start example: Using text prompts with Konwer

This example shows how to use the text-enhanced Konwer model
for training on medical image segmentation with both visual
and text prompts (following the paper's methodology).
"""

import torch
from prompts.visual.load_biomedclip import load_biomedclip
from prompts.text.text_encoder import TextEncoderAdapter
from prompts.text.text_prompt_pipeline import TextPromptPipeline, TextPromptConfig
from models.konwer_sam2d import KonwerSAM2D


def example_1_basic_text_encoding():
    """Example 1: Simple text encoding"""
    print("\n" + "="*60)
    print("Example 1: Basic Text Encoding")
    print("="*60)
    
    # Load BiomedCLIP
    clip_model, _, tokenizer = load_biomedclip(device="cuda")
    text_encoder = TextEncoderAdapter(
        model=clip_model,
        tokenizer=tokenizer,
        device="cuda"
    )
    
    # Encode some medical text
    texts = [
        "malignant tumor in left breast",
        "benign cyst with clear margins",
        "irregular lesion with high vascularity",
    ]
    
    embeddings = text_encoder.encode(texts)
    print(f"✓ Encoded {len(texts)} texts")
    print(f"  Shape: {embeddings.shape}")  # [3, 512]
    print(f"  Embeddings are L2-normalized: {embeddings[0].norm().item():.4f}")


def example_2_text_prompt_generation():
    """Example 2: Generate text prompts with VQA"""
    print("\n" + "="*60)
    print("Example 2: Text Prompt Generation (VQA only)")
    print("="*60)
    
    # Create dummy image batch
    images = torch.randn(2, 3, 256, 256).clamp(0, 1)  # [B,3,H,W] in [0,1]
    
    # Initialize text pipeline (VQA only, no GPT)
    text_cfg = TextPromptConfig(
        vqa_enabled=True,
        gpt_enabled=False,
    )
    text_pipeline = TextPromptPipeline(text_cfg, device="cuda")
    
    # Generate text prompts
    tp = text_pipeline(images, labels=None)
    print(f"✓ Generated {len(tp.text)} text prompts")
    for i, text in enumerate(tp.text[:2]):
        print(f"  [{i}] {text[:60]}..." if len(text) > 60 else f"  [{i}] {text}")


def example_3_model_with_text():
    """Example 3: Model forward pass with text prompts"""
    print("\n" + "="*60)
    print("Example 3: Model with Text Prompts (requires SAM checkpoint)")
    print("="*60)
    
    try:
        from models.load_sam_med2d import load_sam_model
        from core.types import VisualPrompts, TextPrompts
        
        # Create dummy inputs
        images = torch.randn(2, 3, 256, 256).clamp(0, 1)  # [B,3,H,W]
        
        # Dummy visual prompts
        vp = VisualPrompts(
            boxes_xyxy=torch.tensor([
                [[50, 50, 150, 150]],
                [[60, 60, 160, 160]]
            ], dtype=torch.float32),
            points_xy=torch.tensor([
                [[100, 100], [120, 120]],
                [[110, 110], [130, 130]]
            ], dtype=torch.float32),
            points_labels=torch.tensor([
                [1, 1],
                [1, 1]
            ], dtype=torch.long),
        )
        
        # Dummy text prompts
        tp = TextPrompts(
            text=[
                "left breast tumor with irregular shape",
                "right breast lesion with smooth margins"
            ]
        )
        
        # Load SAM
        sam = load_sam_model(
            checkpoint_path="checkpoints/sam_med2d_vit_b.pth",
            model_type="vit_b",
            device="cuda",
        )
        
        # Load BiomedCLIP for text encoding
        clip_model, _, tokenizer = load_biomedclip(device="cuda")
        text_encoder = TextEncoderAdapter(
            model=clip_model,
            tokenizer=tokenizer,
            device="cuda"
        )
        
        # Create model with text support
        model = KonwerSAM2D(
            sam_model=sam,
            text_encoder=text_encoder,
            fusion_mode="concat"  # Try "weighted_sum" or "attention"
        ).to("cuda").eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(images.to("cuda"), vp=vp, tp=tp)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {outputs.mask_logits.shape}")  # [2,1,256,256]
        
    except FileNotFoundError:
        print("✗ SAM checkpoint not found. Skipping forward pass demo.")
        print("  To run this example, download checkpoints/sam_med2d_vit_b.pth")


def example_4_fusion_modes():
    """Example 4: Comparing different fusion modes"""
    print("\n" + "="*60)
    print("Example 4: Fusion Modes Comparison")
    print("="*60)
    
    print("\nAvailable fusion modes:")
    print("  1. 'concat' (default)")
    print("     - Concatenates text [B,512] with visual [B,256]")
    print("     - Projects to [B,256] via Linear layer")
    print("     - Learnable params: Linear(768 → 256)")
    print("     - Best for: Strong text-visual interaction")
    
    print("\n  2. 'weighted_sum'")
    print("     - Learnable weight α")
    print("     - fused = (1-α)·visual + α·text_projected")
    print("     - Learnable params: α + Linear(512 → 256)")
    print("     - Best for: Balanced blending of modalities")
    
    print("\n  3. 'attention'")
    print("     - Treats text as separate prompt token")
    print("     - Text added to point/box prompts")
    print("     - Learnable params: Linear(512 → 256)")
    print("     - Best for: Modular prompt design")
    
    print("\nUsage:")
    print("  model = KonwerSAM2D(sam, text_encoder, fusion_mode='concat')")


def example_5_training_workflow():
    """Example 5: Complete training workflow"""
    print("\n" + "="*60)
    print("Example 5: Training Workflow Overview")
    print("="*60)
    
    code_example = '''
# (1) Initialize components
clip_model, _, tokenizer = load_biomedclip(device="cuda")
text_encoder = TextEncoderAdapter(model=clip_model, tokenizer=tokenizer, device="cuda")
text_pipeline = TextPromptPipeline(TextPromptConfig(vqa_enabled=True), device="cuda")

# (2) Load SAM and create model with text support
sam = load_sam_model("checkpoints/sam_med2d_vit_b.pth", device="cuda")
model = KonwerSAM2D(sam, text_encoder=text_encoder, fusion_mode="concat").to("cuda")

# (3) Training loop
for epoch in range(num_epochs):
    for images, masks, labels in train_dataloader:
        # Generate prompts
        vp = visual_pipeline(images)      # boxes + points from CAM
        tp = text_pipeline(images, labels)  # VQA + GPT text
        
        # Forward pass with BOTH visual and text prompts
        outputs = model(images.to("cuda"), vp=vp, tp=tp)
        loss = criterion(outputs.mask_logits, masks.to("cuda"))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation (same code)
    with torch.no_grad():
        for images, masks, labels in val_dataloader:
            vp = visual_pipeline(images)
            tp = text_pipeline(images, labels)
            outputs = model(images.to("cuda"), vp=vp, tp=tp)
            # Compute metrics...
'''
    print(code_example)


def example_6_comparison_without_text():
    """Example 6: Backward compatibility - model without text"""
    print("\n" + "="*60)
    print("Example 6: Backward Compatibility (Model without Text)")
    print("="*60)
    
    print("The model still works without text prompts:")
    print("\n# Old API (still supported):")
    print("  model = KonwerSAM2D(sam)  # No text_encoder")
    print("  out = model(images, vp)  # No text prompts")
    print("\n# New API (enhanced):")
    print("  model = KonwerSAM2D(sam, text_encoder=text_encoder)")
    print("  out = model(images, vp=vp, tp=tp)  # With text")
    print("\n✓ Both APIs work - choose what's best for your application!")


if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║  KONWER TEXT PROMPTS - QUICK START EXAMPLES               ║")
    print("║  Paper: Enhancing SAM with Efficient Prompting & PO       ║")
    print("║         for Semi-supervised Medical Image Segmentation    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Run examples
    example_1_basic_text_encoding()
    example_2_text_prompt_generation()
    example_3_model_with_text()
    example_4_fusion_modes()
    example_5_training_workflow()
    example_6_comparison_without_text()
    
    print("\n" + "="*60)
    print("For more details, see TEXT_PROMPTS_GUIDE.md")
    print("="*60 + "\n")
