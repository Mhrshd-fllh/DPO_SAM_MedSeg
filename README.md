# Konwer (CVPR 2025) — Paper-faithful 2D Implementation (no DPO)

This repository is a **modular, paper-faithful** PyTorch implementation of the CVPR 2025 paper:

**Konwer: Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation**

> This repo focuses on the **full Konwer pipeline except DPO** (Stage 2).  
> The goal is to implement the paper’s **unsupervised prompting** and **Stage-1 prompt fine-tuning** as accurately as possible,
> while keeping the code **clean, hackable, and easy to extend**.

---

## ✨ Key Features

- **2D medical segmentation**
- **Paper-faithful Unsupervised Prompting**
  - **Visual prompts**: BiomedCLIP → gScoreCAM → DenseCRF → largest CC → **box + points**
  - **Text prompts**: MedVInT-style VQA answer + GPT-4 generic description → concatenated text prompt
- **SAM-style prompt encoder + mask decoder wiring**
- **Stage 1 training** exactly as in the paper:
  - 10% labeled data
  - 15 epochs
  - Adam optimizer
  - lr = 1e-4, halved every 10 epochs
  - resize to 256×256
  - loss = Dice + Focal with weight ratio 20:1
  - use **box + points** together

---

## 📦 Repo structure

```txt
konwer2d/
  configs/
    default.yaml
    datasets.yaml
    prompts.yaml

  core/
    types.py
    registry.py
    config.py

  data/
    datasets/
      busi_dataset.py
    transforms/
      resize_sam_med2d.py

  prompts/
    visual/
      biomedclip_gscorecam.py
      densecrf.py
      postprocess.py
      visual_prompt_pipeline.py
    text/
      vqa_medvint_adapter.py
      gpt4_adapter.py
      text_prompt_pipeline.py
    prompt_bundle.py

  models/
    backbones/
      sam_med2d_adapter.py
    prompt_encoder/
      sam_prompt_encoder_adapter.py
    mask_decoder/
      sam_mask_decoder_adapter.py
      mask_feature_fuser.py
    konwer_model.py

  losses/
    dice.py
    focal.py
    sup_combo.py

  train/
    stage1_prompt_finetune.py
    engine.py
    evaluate.py

  scripts/
    train_stage1.py
    eval.py

  requirements.txt
  README.md
```


## 🧠 Dataset Choice (Lightweight 2D)

This repo is designed for the **BUSI breast ultrasound segmentation dataset** (lightweight & fast).

The paper uses:
- **BUSI + UDIAT**: total **810 images**
- **600 train / 210 test**

## 📁 Dataset format (expected)

Prepare your dataset like this:

```txt
data/BUSI/
  train/
    images/
      0001.png
      0002.png
      ...
    masks/
      0001.png
      0002.png
      ...
  test/
    images/
      0001.png
      0002.png
      ...
    masks/
      0001.png
      0002.png
      ...
```

### Notes
- Masks must be single-channel
- Any non-zero value is treated as foreground
- Image and mask filenames must match

## 🚀 Quick Start (Colab)

### 1) Install Dependencies
```bash
pip install -r requirements.txt
```

### 2) Stage 1-Prompt Fine-Tuning (Paper Faithful)
```bash
python scripts/train_stage1.py \
  --data_root data/BUSI \
  --out_dir runs/busi \
  --config configs/default.yaml
```

## 🧩 Unsupervised Prompting (Paper-faithful)

### Visual prompt pipeline
For each image:
1. Encode image + label text using BiomedCLIP
2. Generate saliency map using gScoreCAM
3. Refine mask with DenseCRF
4. Keep the largest connected component (area constraint)
5. Convert mask → bounding box + random points

Outputs:
- `boxes_xyxy`: `[B, 1, 4]`
- `points_xy`: `[B, K, 2]`
- `points_labels`: `[B, K]`

### Text prompt pipeline
For each image:
1. Run VQA (MedVInT-style) using template:
  ```yaml
  Question: {}, Answer is:
  ``` 
2. Quert GPT-4 for a generic description based on the organ/disease label
3. Concatenate:
  ```ini
  final_text = vqa_answer + " " + gpt_description
  ```

Outputs:
- `text`: list of strings (length = batch size)

## 🏗 Model Design Philosophy
The repo is intentionally written in a highly modular style:
- You can swap any component independently:
  - Image encoder (SAM-Med2D / MedSAM)
  - Text encoder (BiomedCLIP text encoder)
  - Prompt encoder (SAM prompt encoder)
  - Mask encoder 
  - Visual prompt generator
  - Text prompt generator

This makes it easy to:
- combine two ideas
- run ablations
- test alternative prompt strategies
- upgrade backbones later

## ⚠️ About DPO (Stage 2)

The original paper includes Stage 2 (preference alignment via a DPO-style objective).  
This repo intentionally **does not implement DPO**.

The goal here is:
- to implement Stage 1 and unsupervised prompting **as accurately as possible**
- to keep the codebase clean and extensible for future experimentation

---

## 🔧 Configuration

Configs are stored in:
- `configs/default.yaml`
- `configs/prompts.yaml`
- `configs/datasets.yaml`

Key hyperparameters:
- `img_size`: 256
- `epochs`: 15
- `lr`: 1e-4
- `lr_step`: 10 epochs
- `loss`: Dice + Focal (20:1)
- `num_points`: number of sampled point prompts
- `thresholds`: for CRF mask post-processing

---

## 🛠 Troubleshooting

### Training is slow in Colab
- reduce batch size (e.g. 4)
- set `num_workers=0`

### Masks are empty
- verify your mask files are not fully black
- ensure masks use non-zero pixels for foreground

### Visual prompting is unstable
- check DenseCRF parameters
- check that saliency maps are not saturated
- verify connected-component filtering

---

## 📌 Roadmap (Recommended Next Steps)

- Add SAM-Med2D weights + real prompt encoder/decoder wrappers
- Replace VQA stub with real MedVInT checkpoints
- Replace GPT stub with OpenAI API backend
- Add DPO stage (optional)

---

## License
MIT
