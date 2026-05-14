# univad-reproduced

Clean reproduction of UniVAD (Gu et al., CVPR 2025) on MVTec-AD, VisA, MVTec LOCO, and BMAD. A training-free unified few-shot visual anomaly detection across industrial, logical, and medical domains.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Paper:** UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection --- Gu et al., CVPR 2025 · [arXiv:2412.03342](https://arxiv.org/abs/2412.03342)

## What this is

End-to-end reproduction of UniVAD, the first model that is simultaneously source-free, few-shot, and unified across industrial, logical, and medical anomaly detection. Prior unified models like UniAD require large amounts of normal training data per category. Prior few-shot detectors like AnomalyDINO and WinCLIP cover only industrial benchmarks. UniVAD removes both constraints: a frozen DINOv2/CLIP backbone paired with a handful of normal reference images handles all three domains without any offline training on the target domain.

The model operates through three modules. C³ (Contextual Component Clustering) combines Grounded SAM with K-means clustering to segment object components under few-shot conditions. CAPM (Component-Aware Patch Matching) restricts patch-level feature matching to within-component regions, eliminating false positives from background and irrelevant regions. GECM (Graph-Enhanced Component Modeling) builds a graph over component features and uses geometric and deep features jointly to catch logical anomalies, missing parts, wrong colors, and incorrect counts that patch matching alone cannot detect.

This is the fifth reproduction in a series covering UniVAD's full comparison set. UniVAD is the current state-of-the-art training-free unified VAD model and the natural endpoint of this series. Every earlier reproduction (PatchCore, WinCLIP, UniAD, MedCLIP) is a method UniVAD directly compares against in the paper. Getting clean numbers on all of them in one place makes the comparison meaningful. The remaining baselines from the paper will be added to the series once UniVAD reproduction is complete.

## Status

Industrial datasets prepared and indexed (MVTec-AD, VisA in 1cls format, MVTec LOCO with merged Caption masks). meta.json files generated for all three. Pretrained checkpoints in place. Medical datasets queued for download from the BMAD OneDrive. Component segmentation pre-computation and baseline runs pending.

Results will land in the table below as experiments complete.

## Goal

Match the paper's reported 1-shot numbers within ±0.5 points across all six evaluation datasets:

| Domain | Dataset | Image-AUROC (Paper) | Image-AUROC (This repo) | Pixel-AUROC (Paper) | Pixel-AUROC (This repo) |
|--------|---------|--------------------|-----------------------|--------------------|-----------------------|
| Industrial | MVTec-AD | 97.8 | TBD | 96.5 | TBD |
| Industrial | VisA | 93.5 | TBD | 98.2 | TBD |
| Logical | MVTec LOCO | 71.0 | TBD | 75.1 | TBD |
| Medical | BrainMRI | 80.2 | TBD | 96.8 | TBD |
| Medical | LiverCT | 70.0 | TBD | 96.3 | TBD |
| Medical | RESC | 85.5 | TBD | 94.9 | TBD |

All evaluations use the **1-normal-shot** setting: one normal reference image per category, no training on target domain data.

## Why UniVAD

The standard approach to visual anomaly detection trains one model per object category from hundreds of normal samples. That works on benchmarks. It does not work in a factory with 200 product types, a hospital with dozens of imaging protocols, or any deployment where the target distribution is not known in advance.

UniVAD is the paper that breaks this constraint cleanly. One frozen backbone. A few reference images. Any domain. No retraining.

For this series specifically: UniVAD's freeze-and-infer design is also its main limitation. It assumes the deployment distribution stays fixed after the reference images are indexed. In practice, new product lines arrive, scanners get upgraded, patient cohorts shift. The model has no mechanism to adjust. This reproduction measures exactly how much performance degrades under that assumption, the numbers here are the lower bound that any continual-adaptation extension is built to improve.

## Installation

```bash
git clone --recurse-submodules https://github.com/hammadhaideer/univad-reproduced.git
cd univad-reproduced
conda env create -f environment.yml
conda activate univad
```

Install GroundingDINO (vendored as a submodule):

```bash
cd models/GroundingDINO
pip install -e . --no-build-isolation
cd ../..
```

Clone the DINOv2 model definition (required, weights do not download automatically):

```bash
git clone https://github.com/facebookresearch/dinov2.git models/dinov2
```

CLIP weights download automatically on first run via OpenCLIP.

## Pretrained Checkpoints

Download both checkpoints to `pretrained_ckpts/` before running:

```bash
cd pretrained_ckpts

# SAM-HQ ViT-H (2.6 GB)
curl -L -C - -o sam_hq_vit_h.pth \
  "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"

# GroundingDINO SwinT (694 MB)
curl -L -o groundingdino_swint_ogc.pth \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
```

| Checkpoint | Size | Purpose |
|-----------|------|---------|
| sam_hq_vit_h.pth | 2.6 GB | High-quality component segmentation |
| groundingdino_swint_ogc.pth | 694 MB | Open-set object grounding |

## Datasets

Three of the four dataset groups need a reorganization step before UniVAD can read them. The scripts under `scripts/` handle that.

### MVTec-AD

Download from the [official page](https://www.mvtec.com/company/research/datasets/mvtec-ad), extract to `data/mvtec/`. Then:

```bash
python scripts/mvtec_solver.py
```

Note: UniVAD's upstream `mvtec_solver.py` lists `"metal nut"` (with a space) in `CLSNAMES`, but the official MVTec folder is `metal_nut` (with an underscore). This repo ships the patched version that matches the folder name.

### VisA

Download from [Amazon S3](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) and extract. The raw layout (`<class>/Data/Images/{Anomaly,Normal}`) is not what UniVAD reads. Convert it to the 1-class MVTec-style layout used by spot-diff:

```bash
python scripts/visa_to_1cls.py \
  --data-folder /path/to/VisA_20220922 \
  --save-folder data/VisA_pytorch \
  --split-file /path/to/VisA_20220922/split_csv/1cls.csv
```

This mirrors `amazon-science/spot-diff` `prepare_data.py` for `--split-type 1cls`. The output goes to `data/VisA_pytorch/1cls/`. Then:

```bash
python scripts/visa_solver.py
```

### MVTec LOCO

Download raw [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco), extract to `data/mvtec_loco_caption/`. UniVAD reads the *Caption* variant: a single binary mask per defect image, rather than the multiple component masks the raw release ships. Build the merged masks in place:

```bash
python scripts/loco_merge.py --root data/mvtec_loco_caption
```

The script OR-merges per-component masks for each defect image and writes them to `<class>/ground_truth_merge_mask/<defect>_merge_mask/<id>.png`. It mirrors `hujiecpp/MVTec-Caption` `Construct_MVTEC-LOCO-Caption.py`. Then:

```bash
python scripts/mvtec_loco_solver.py
```

### BMAD (Medical)

Download from the BMAD-derived [OneDrive](https://1drv.ms/u/s!AopsN_HMhJeckoJT-3yF_pwQMSn9OA?e=nRW1wA) provided by the UniVAD authors and extract to `data/`. The pack is BMAD pre-organized in MVTec-style layout. UniVAD's `data/` structure expects six folders with exact case-sensitive names: `BrainMRI`, `LiverCT`, `RESC`, `HIS`, `ChestXray`, `OCT17`. No solver script is needed — `meta.json` ships with the pack for each dataset.

### Expected layout

```bash
data/
├── mvtec/
│   ├── meta.json
│   ├── bottle/
│   └── ... (15 categories)
├── VisA_pytorch/1cls/
│   ├── meta.json
│   └── ... (12 categories)
├── mvtec_loco_caption/
│   ├── meta.json
│   └── ... (5 categories, each with ground_truth/ + ground_truth_merge_mask/)
├── BrainMRI/
│   ├── meta.json
│   ├── train/
│   ├── test/
│   └── ground_truth/
├── LiverCT/
├── RESC/
├── HIS/
├── ChestXray/
└── OCT17/
```

## Run

Pre-compute component segmentation masks for all datasets before evaluation:

```bash
python scripts/segment_components.py
```

Run full evaluation:

```bash
bash scripts/test.sh
```

Or evaluate a single dataset:

```bash
python scripts/test_univad.py --dataset mvtec --shot 1
python scripts/test_univad.py --dataset visa --shot 1
python scripts/test_univad.py --dataset mvtec_loco --shot 1
python scripts/test_univad.py --dataset brainmri --shot 1
```

## Roadmap

- [x] Environment setup — torch 2.0.1, GroundingDINO, SAM-HQ
- [x] Pretrained checkpoints — GroundingDINO SwinT, SAM-HQ ViT-H
- [x] Industrial dataset preparation — MVTec-AD, VisA (1cls), MVTec LOCO Caption
- [x] meta.json indexing for industrial datasets
- [ ] Medical dataset preparation — BMAD (6 sets)
- [ ] Component segmentation pre-computation
- [ ] Baseline reproduction — MVTec-AD 1-shot
- [ ] Baseline reproduction — VisA, MVTec LOCO, BMAD
- [ ] Sequential shift motivation experiment (S1/S2/S3 protocols)
- [ ] Results table complete
- [ ] Walkthrough notebook

## Reproduction series

- [x] [patchcore-reproduced](https://github.com/hammadhaideer/patchcore-reproduced) --- PatchCore (CVPR 2022)
- [x] [winclip-reproduced](https://github.com/hammadhaideer/winclip-reproduced) --- WinCLIP (CVPR 2023)
- [x] [uniad-reproduced](https://github.com/hammadhaideer/uniad-reproduced) --- UniAD (NeurIPS 2022)
- [x] [medclip-reproduced](https://github.com/hammadhaideer/medclip-reproduced) --- MedCLIP (EMNLP 2022)
- [ ] **univad-reproduced** — UniVAD (CVPR 2025) ← this repo

## References

1. Gu et al., UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection, CVPR 2025
2. Bergmann et al., MVTec AD, CVPR 2019
3. Bergmann et al., Beyond Dents and Scratches (MVTec LOCO), IJCV 2022
4. Zou et al., VisA, ECCV 2022
5. Bao et al., BMAD: Benchmarks for Medical Anomaly Detection, CVPR Workshops 2024
6. Kirillov et al., Segment Anything, ICCV 2023
7. Liu et al., Grounding DINO, arXiv 2023

## License

MIT
