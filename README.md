<div align="center">

# Neuro-MRI

**AI-Assisted Brain MRI Screening Console**

An ensemble deep-learning framework that classifies T1-weighted contrast-enhanced
(T1W-CE) brain MRI scans into **glioma**, **meningioma**, **pituitary tumor**, and
**no tumor** — powered by seven deep-learning models, delivered as a production-grade
web application.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Tracked-F05133?style=flat-square&logo=gitlfs&logoColor=white)

**Research / coursework tool — not a diagnostic device.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [The Seven Models](#the-seven-models)
- [Benchmark Results](#benchmark-results)
- [Ensemble Strategy](#ensemble-strategy)
- [Web Application](#web-application)
- [REST API](#rest-api)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training the Models](#training-the-models)
- [Security & Privacy](#security--privacy)
- [Accessibility & PWA](#accessibility--pwa)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)
- [Acknowledgments](#acknowledgments)

---

## Overview

Brain tumors are among the most serious neurological conditions, and early detection
directly improves treatment outcomes. MRI is the gold-standard imaging modality, but
its interpretation depends on expert radiologists — a scarce resource in many parts
of the world.

**Neuro-MRI** provides an automated, reproducible, second-opinion screening system:

1. Upload a T1W-CE brain MRI scan (drag & drop, or pick from built-in demo samples).
2. Seven independent deep-learning models analyze the image simultaneously.
3. Their opinions are combined into one transparent verdict using three aggregation
   strategies — **majority vote**, **probability average**, and the production
   **accuracy-weighted ensemble** — with a per-model breakdown and confidence scores.

Everything runs in memory: scans are never stored, uploaded to third parties, or used
to identify users.

---

## Features

- **7-model ensemble** — CNN, Vision Transformer, Hybrid CNN–ViT, ResNet50 (×2),
  DenseNet121 (×2) across three training paradigms (from scratch, frozen pretrained,
  fine-tuned).
- **Zero-overlap validation** — automated check confirming no leakage between
  train/validation/test, so benchmark numbers reflect genuine generalization.
- **Accuracy-weighted voting** — model weight = test accuracy⁴, blended as
  70% weighted probability + 30% weighted vote.
- **Production-grade UI** — glassmorphic design, dark/light themes, drag-and-drop
  upload, shimmer skeleton loaders, toast notifications, scroll-progress bar.
- **PWA-ready** — `manifest.json`, service worker, SVG favicon; installable and
  offline-capable.
- **Accessibility-first** — skip links, ARIA labels, focus management, keyboard
  navigation, `prefers-reduced-motion` support.
- **REST API** — machine-readable predictions via `POST /api/predict`.
- **Fast cold start** — models load lazily in a thread-safe registry with a startup
  warm-up pass, so the first request doesn't pay the full load cost.

---

## Dataset

| Property      | Value                                                        |
| ------------- | ------------------------------------------------------------ |
| Source        | Epic & CSCR Hospital, Bangladesh (real clinical scans)       |
| Modality      | T1-weighted contrast-enhanced (T1W-CE) MRI                   |
| Total images  | 5,941                                                         |
| Classes       | Glioma, Meningioma, Pituitary tumor, No tumor                |
| Split         | Stratified 70 / 15 / 15 (train 4,160 · val 892 · test 889)  |
| Leakage check | Automated zero-overlap verification between all partitions  |

### Class distribution

| Class      | Train | Validation | Test | Total | Share |
| ---------- | ----- | ---------- | ---- | ----- | ----- |
| Glioma     | 1,534 | 329        | 328  | 2,191 | 36.9% |
| Meningioma | 1,035 | 222        | 221  | 1,478 | 24.9% |
| Pituitary  | 1,040 | 223        | 222  | 1,485 | 25.0% |
| No tumor   | 551   | 118        | 118  | 787   | 13.2% |

---

## The Seven Models

| Model                       | Paradigm          | Description                                      |
| --------------------------- | ----------------- | ------------------------------------------------ |
| `DenseNet121_finetuned`     | Fine-tuned        | ImageNet-pretrained DenseNet121, fully fine-tuned on MRI |
| `ResNet50_finetuned`        | Fine-tuned        | ImageNet-pretrained ResNet50, fully fine-tuned on MRI    |
| `ResNet50_frozen`           | Frozen pretrained | ImageNet backbone frozen; classifier head only    |
| `DenseNet121_frozen`        | Frozen pretrained | ImageNet backbone frozen; classifier head only    |
| `CNN_scratch`               | From scratch      | Custom CNN trained from random initialization     |
| `Hybrid_CNN_ViT_scratch`    | From scratch      | CNN feature extractor feeding a Vision Transformer |
| `ViT_scratch`               | From scratch      | Pure patch-based Vision Transformer               |

Model weights are tracked with **Git LFS** (`*.keras`). Run `git lfs pull` after
cloning to download them.

---

## Benchmark Results

Evaluated on 889 scans never seen during training (zeros-overlap test set):

| Rank | Architecture            | Paradigm          | Accuracy | Weighted F1 | Macro F1 |
| ---- | ----------------------- | ----------------- | -------- | ----------- | -------- |
| 1    | DenseNet121_finetuned   | Fine-tuned        | **94.15%** | 94.1%     | 93.8%    |
| 2    | ResNet50_finetuned      | Fine-tuned        | 91.68%   | 91.7%       | 91.2%    |
| 3    | CNN_scratch             | From scratch      | 89.20%   | 89.1%       | 88.5%    |
| 4    | ResNet50_frozen         | Frozen pretrained | 88.98%   | 89.0%       | 88.3%    |
| 5    | DenseNet121_frozen      | Frozen pretrained | 85.49%   | 85.5%       | 84.7%    |
| 6    | Hybrid_CNN_ViT_scratch  | From scratch      | 80.54%   | 80.4%       | 79.2%    |
| 7    | ViT_scratch             | From scratch      | 72.89%   | 72.7%       | 70.1%    |

### Key findings

- **Fine-tuning beats frozen features.** DenseNet121 gains ~8.7 pp (85.49% → 94.15%)
  and ResNet50 ~2.7 pp (88.98% → 91.68%) when ImageNet weights adapt to the MRI
  domain.
- **From-scratch CNNs hold up.** A custom CNN reaches 89.20% on a leakage-free test
  set — and is the fastest model.
- **Transformers need data.** ViT (72.89%) and Hybrid CNN–ViT (80.54%) underperform;
  they typically require 100k+ images to compete with CNNs on ~5.9k images.
- **Hardest cases:** the glioma vs. no-tumor boundary; pituitary and meningioma are
  classified most reliably.

Full per-model metrics and confusion matrices live in
[`brain_tumor_webapp/web_app_assets/`](brain_tumor_webapp/web_app_assets/).

---

## Ensemble Strategy

Each model produces an independent 4-class probability vector. Three methods are
compared and surfaced to the user:

| Strategy                | Formula                                                     | Notes                              |
| ----------------------- | ----------------------------------------------------------- | ---------------------------------- |
| Majority vote           | `argmax(counts)` with avg-confidence tie-break              | Each model votes once              |
| Probability average     | `argmax(mean(P_i(c)))`                                      | Uses all models' uncertainty       |
| **Accuracy-weighted**   | `0.70 · Σ wᵢPᵢ(c) + 0.30 · Σ wᵢ·voteᵢ(c)`, `wᵢ = accᵢ⁴`    | Production default                 |

The exponent of 4 stretches the 0.73–0.94 accuracy spread so a 94% model receives
roughly **3× the weight** of a 73% model — top models dominate the verdict.

---

## Web Application

**Stack:** Flask 3.x · TensorFlow/Keras 2.x · vanilla JavaScript · Jinja2 · PWA

The single-page console (`brain_tumor_webapp/templates/index.html`) features:

- Glassmorphic dark/light UI with an ambient aurora background and reduced-motion support
- Sticky navbar with scroll elevation, active-section scrollspy, and mobile menu
- Drag-and-drop upload with client-side validation (type + 16 MB limit)
- Demo-sample gallery for one-click testing
- Custom loading overlay with step-by-step progress + shimmer skeleton in the result panel
- Verdict panel: weighted score, majority-vote consensus, ensemble result, class
  probability bars, and an expandable per-model table (confidence, weight, latency)
- Toast notifications, back-to-top button, scroll-progress bar

### Run locally

```bash
cd brain_tumor_webapp
pip install -r requirements.txt
python app.py
# → http://127.0.0.1:5000
```

First startup takes ~1 minute while the seven models warm up in the background.

---

## REST API

### `POST /api/predict`

Multipart form with an `image` field → JSON predictions.

**Request**

```bash
curl -X POST -F "image=@scan.jpg" http://127.0.0.1:5000/api/predict
```

**Response**

```json
{
  "class_names": ["glioma", "meningioma", "no_tumor", "pituitary"],
  "models": [
    {
      "name": "DenseNet121_finetuned",
      "label": "glioma",
      "confidence": 0.9421,
      "latency_ms": 48.3,
      "probs": {"glioma": 0.94, "meningioma": 0.02, "no_tumor": 0.01, "pituitary": 0.03}
    }
  ],
  "majority_vote": {"label": "glioma", "confidence": 0.83},
  "ensemble": {"label": "glioma", "confidence": 0.91, "probabilities": {}},
  "weighted": {
    "label": "glioma",
    "confidence": 0.87,
    "scores": {},
    "prob_score": {},
    "vote_score": {},
    "weights": {"DenseNet121_finetuned": 0.21},
    "blend": [0.7, 0.3]
  },
  "votes": {"glioma": 5, "meningioma": 1, "no_tumor": 1}
}
```

Errors are returned as `{"error": "..."}` with appropriate 4xx/5xx status codes.

---

## Project Structure

```
.
├── brain_tumor_colab_loop.py              # End-to-end training loop (Google Colab)
├── brain_tumor_webapp/                    # Production Flask application
│   ├── app.py                             # Flask entry point, inference pipeline, API
│   ├── requirements.txt
│   ├── models/                            # 7 × .keras weights (Git LFS)
│   │   ├── CNN_scratch.keras
│   │   ├── DenseNet121_finetuned.keras
│   │   ├── DenseNet121_frozen.keras
│   │   ├── Hybrid_CNN_ViT_scratch.keras
│   │   ├── ResNet50_finetuned.keras
│   │   ├── ResNet50_frozen.keras
│   │   └── ViT_scratch.keras
│   ├── static/
│   │   ├── css/style.css                  # Full design system (~900 lines)
│   │   ├── js/script.js                   # Theme, nav, AJAX, PWA, toasts
│   │   ├── favicon.svg · manifest.json · sw.js
│   │   └── img/samples/                   # 4 demo MRI scans
│   ├── templates/index.html               # Single-page console (Jinja2)
│   └── web_app_assets/                    # Benchmark CSVs + model manifest
├── .gitattributes                         # Git LFS tracking (*.keras)
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- *(optional)* Git LFS — `git lfs install` before cloning, or run `git lfs pull`

### 1. Clone

```bash
git clone https://github.com/joyjitbhowal/Neuro-MRI.git
cd Neuro-MRI
git lfs pull        # downloads the 7 model weights (~660 MB)
```

### 2. Install & run

```bash
cd brain_tumor_webapp
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> and upload a scan or click a demo sample.

> Building from source with TensorFlow for Linux: `pip install tensorflow`
> (the models were trained on TF 2.x; `requirements.txt` covers the webapp runtime).

---

## Training the Models

[`brain_tumor_colab_loop.py`](brain_tumor_colab_loop.py) is the complete training
pipeline used to produce all seven models. It assumes the dataset is mounted in
Google Drive (folder structure `train/val/test` with one subfolder per class) and
drives **TensorFlow + Keras** with:

- Per-architecture preprocessing (ImageNet normalization for pretrained backbones,
  1/255 rescaling for from-scratch models)
- Adam optimizer with cosine-annealed learning rate and early stopping
- Class-weighting for the imbalanced no-tumor class
- Stratified 70/15/15 split with **automated zero-overlap verification**
- Model checkpoints saved as `.keras` into the same layout expected by the webapp

Run it in a Colab notebook (or locally with the same structure) and drop the
resulting `.keras` files into `brain_tumor_webapp/models/`.

---

## Security & Privacy

- Images are processed **in memory only** — never written to disk, never uploaded
  to any cloud service, never used for any purpose beyond the single analysis.
- No accounts, logins, or personal information required.
- Defense-in-depth HTTP headers on every response:
  `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: strict-origin-when-cross-origin`, and a restrictive
  `Permissions-Policy`.
- Upload cap of 16 MB enforced both client-side and server-side.

---

## Accessibility & PWA

- WCAG-minded: skip-to-content link, ARIA labels, `aria-current` on active nav,
  focus management on results, `prefers-reduced-motion` support.
- Keyboard friendly: fully operable menus and forms without a mouse.
- Installable via the web manifest; a network-first service worker keeps the UI
  available offline and caches static assets (API responses are never cached).

---

## Limitations

- Single acquisition center and single imaging sequence (T1W-CE).
- Only three tumor types + no-tumor category; other brain conditions are not covered.
- Classification only — no tumor segmentation or localization.
- Transformer variants are data-hungry and underperform at this dataset size.
- Accuracy should **not** be conflated with clinical reliability; no prospective
  clinical validation has been performed.

---

## Future Work

- Add a segmentation head (e.g., U-Net) for tumor localization
- Fuse multiple MRI sequences (T2, FLAIR, DWI)
- Multi-center validation and federated training across hospitals
- Uncertainty quantification via Monte-Carlo dropout
- GPU-based inference with batched requests for higher throughput

---

## Disclaimer

**Neuro-MRI is a screening and educational tool, not a medical device.** It does not
replace a doctor, radiologist, imaging test, or biopsy, and it is not approved for
clinical use. Always discuss your results with a qualified healthcare professional
before making any decision about your health.

---

## Acknowledgments

- Dataset: Epic & CSCR Hospital, Bangladesh
- Supervised under the **Technology Innovation Hub (TIH), Indian Institute of
  Technology** internship programme
- Built with TensorFlow/Keras · Flask · Jinja2 · vanilla JavaScript