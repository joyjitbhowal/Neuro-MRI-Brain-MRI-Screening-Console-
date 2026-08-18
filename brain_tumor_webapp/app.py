"""BDNeuro-MRI web app - single-page scan console.

Sections (anchor nav): hero (#home), about (#about), benchmark (#benchmark),
analysis (#analysis), classification (#classification).

Routes:
  GET  /                  single page (optional ?sample=<file> pre-runs prediction)
  POST /                  upload an MRI -> prediction rendered on the same page
  POST /api/predict       JSON API (multipart 'image') - same pipeline
"""

import os
import io
import json
import uuid
import time
import threading
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from flask import (Flask, render_template, request, url_for, flash,
                   redirect, jsonify)

import tensorflow as tf
from tensorflow.keras import applications

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "web_app_assets")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
GEN_DIR    = os.path.join(BASE_DIR, "static", "generated")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GEN_DIR,    exist_ok=True)

with open(os.path.join(ASSETS_DIR, "model_manifest.json")) as f:
    MANIFEST = json.load(f)

CLASS_NAMES  = MANIFEST["class_names"]
IMG_SIZE     = tuple(MANIFEST["img_size"])
ALLOWED_EXTS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# per-class accent chips (semantic class colors, kept from the product)
CLASS_COLORS = {"glioma": "#e63946", "meningioma": "#457b9d",
                "no_tumor": "#2a9d8f", "pituitary": "#e9a23b"}

# training paradigm per architecture (the variable being compared)
PARADIGMS = {
    "DenseNet121_finetuned":    "Fine-tuned",
    "ResNet50_finetuned":       "Fine-tuned",
    "ResNet50_frozen":          "Frozen pretrained",
    "DenseNet121_frozen":       "Frozen pretrained",
    "CNN_scratch":              "From scratch",
    "Hybrid_CNN_ViT_scratch":   "From scratch",
    "ViT_scratch":              "From scratch",
}

# per-model scoring weights = test-set accuracy (source: per_model_metrics.csv)
MODEL_WEIGHTS = {}
_metrics_path = os.path.join(ASSETS_DIR, "per_model_metrics.csv")
if os.path.exists(_metrics_path):
    _mdf = pd.read_csv(_metrics_path)
    MODEL_WEIGHTS = dict(zip(_mdf["Architecture"], _mdf["Test_Accuracy"]))
_FALLBACK_WEIGHTS = {
    "DenseNet121_finetuned": 0.9415, "ResNet50_finetuned": 0.9168,
    "CNN_scratch": 0.8920, "ResNet50_frozen": 0.8898,
    "DenseNet121_frozen": 0.8549, "Hybrid_CNN_ViT_scratch": 0.8054,
    "ViT_scratch": 0.7289,
}
for _k, _v in _FALLBACK_WEIGHTS.items():
    MODEL_WEIGHTS.setdefault(_k, _v)

app = Flask(__name__)
app.secret_key = "bdneuro-mri-webapp"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600      # static assets: 1 h cache

@app.after_request
def security_headers(resp):
    """Defense-in-depth headers for production serving."""
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy",
                            "camera=(), microphone=(), geolocation=(), payment=()")
    return resp

@app.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.svg"))

# ----------------------------------------------------------------- model serving
_models = {}
_models_lock = threading.Lock()

def get_model(name):
    with _models_lock:
        if name not in _models:
            path = os.path.join(MODELS_DIR, f"{name}.keras")
            if not os.path.exists(path):
                raise FileNotFoundError(f"model not present: {name}")
            _models[name] = tf.keras.models.load_model(path)
        return _models[name]

def preprocess(x, mode):
    """x: float32 numpy array (1, H, W, 3) in [0, 255]. Same rules as training.
    NOTE: keras preprocess_input() mutates its input in place, so copy first."""
    x = x.copy()
    if mode == "rescale_255":
        return x / 255.0
    if mode == "resnet50":
        return applications.resnet50.preprocess_input(x)
    if mode == "densenet121":
        return applications.densenet.preprocess_input(x)
    raise ValueError(f"unknown preprocessing mode: {mode}")

def predict_all(pil_img):
    """Run one image through every available model. Returns per-model results,
    majority-vote consensus, and probability-average ensemble."""
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.asarray(img, dtype="float32")[None, ...]          # (1, H, W, 3)

    results = []
    for name, info in MANIFEST["models"].items():
        weights = os.path.join(MODELS_DIR, info["file"].replace("models/", ""))
        if not os.path.exists(weights):
            continue
        t0 = time.perf_counter()
        probs = get_model(name).predict(preprocess(x, info["preprocessing"]),
                                        verbose=0)[0]
        latency = (time.perf_counter() - t0) * 1000.0
        top = int(np.argmax(probs))
        results.append({
            "name":      name,
            "label":     CLASS_NAMES[top],
            "confidence": float(probs[top]),
            "latency_ms": round(latency, 1),
            "probs":     {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        })

    # ---- majority vote (ties broken by average confidence) ----
    votes = Counter(r["label"] for r in results)
    by_label = {}
    for r in results:
        by_label.setdefault(r["label"], []).append(r["confidence"])
    consensus_label = max(by_label,
                          key=lambda l: (votes[l],
                                         sum(by_label[l]) / len(by_label[l])))
    consensus_conf = sum(by_label[consensus_label]) / len(by_label[consensus_label])

    # ---- average-probability ensemble ----
    avg = np.zeros(len(CLASS_NAMES))
    for r in results:
        avg += np.array([r["probs"][c] for c in CLASS_NAMES])
    avg /= len(results)
    ei = int(np.argmax(avg))

    return (results, consensus_label, float(consensus_conf),
            CLASS_NAMES[ei], float(avg[ei]), avg,
            sorted(votes.items(), key=lambda kv: -kv[1]))

# ------------------------------------------------------------------- scoring
# exponent applied to test accuracy when deriving model weights: accuracy^4
# stretches the spread so top models dominate instead of all weights being
# nearly equal (plain accuracy differs only ~0.73-0.94).
WEIGHT_POWER = 4

def weighted_scoring(results):
    """Accuracy-weighted scoring ensemble.

    Every model contributes with weight = test_accuracy ** WEIGHT_POWER
    (normalized to sum to 1), so stronger models dominate the verdict:

      prob_score(c)  = sum_i w_i * P_i(c)              # weighted probability
      vote_score(c)  = sum_i w_i * [c == argmax_i]     # weighted vote
      score(c)       = 0.70 * prob_score(c) + 0.30 * vote_score(c)

    The verdict is argmax(score); confidence = that class's share of the
    total score mass.
    """
    ws = [MODEL_WEIGHTS.get(r["name"], 0.5) ** WEIGHT_POWER for r in results]
    total_w = sum(ws) or 1.0
    ws = [w / total_w for w in ws]

    prob = {c: 0.0 for c in CLASS_NAMES}
    vote = {c: 0.0 for c in CLASS_NAMES}
    for r, w in zip(results, ws):
        for c in CLASS_NAMES:
            prob[c] += w * r["probs"][c]
        vote[r["label"]] += w

    score = {c: 0.70 * prob[c] + 0.30 * vote[c] for c in CLASS_NAMES}
    label = max(score, key=score.get)
    return {
        "label":       label,
        "confidence":  score[label],
        "prob_score":  {c: float(v) for c, v in prob.items()},
        "vote_score":  {c: float(v) for c, v in vote.items()},
        "scores":      {c: float(v) for c, v in score.items()},
        "weights":     {r["name"]: round(w, 4) for r, w in zip(results, ws)},
        "blend":       (0.70, 0.30),
    }

# ------------------------------------------------------------------- image utils
def save_thumbnail(pil_img):
    """Small preview copy for the result panel; returns public URL."""
    thumb = pil_img.convert("RGB")
    thumb.thumbnail((256, 256))
    fname = f"{uuid.uuid4().hex}.jpg"
    thumb.save(os.path.join(GEN_DIR, fname), "JPEG", quality=88)
    return url_for("static", filename=f"generated/{fname}")

def load_image(file_storage):
    """Validate + open an uploaded image (by memory to keep it simple & safe)."""
    raw = file_storage.read()
    if not raw:
        raise ValueError("Empty file.")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise ValueError("File is not a readable image (allow .png/.jpg/.jpeg/.bmp/.tif).")
    return img.convert("RGB")

def finalize(pil_img):
    """Shared result pipeline used by the UI."""
    results, con_label, con_conf, ens_label, ens_conf, avg, votes = predict_all(pil_img)
    con_votes = dict(votes).get(con_label, 0)
    w = weighted_scoring(results)
    return {
        "img_url":   save_thumbnail(pil_img),
        "results":   results,
        "con_label": con_label,
        "con_conf":  con_conf,
        "con_votes": con_votes,
        "ens_label": ens_label,
        "ens_conf":  ens_conf,
        "avg":       {c: float(p) for c, p in zip(CLASS_NAMES, avg)},
        "votes":     votes,
        "n_models":  len(results),
        "w_label":   w["label"],
        "w_conf":    w["confidence"],
        "w_scores":  w["scores"],
        "w_weights": w["weights"],
        "w_blend":   w["blend"],
    }

# ------------------------------------------------------------------ site data
def benchmark_data():
    """Rows from model_benchmark_results.csv, sorted by accuracy desc."""
    path = os.path.join(ASSETS_DIR, "model_benchmark_results.csv")
    if not os.path.exists(path):
        return [], []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.sort_values("Test_Accuracy", ascending=False).iterrows():
        arch = r["Architecture"]
        rows.append({
            "arch":     arch,
            "paradigm": PARADIGMS.get(arch, "-"),
            "accuracy": r["Test_Accuracy"] * 100,
            "f1":       r["Test_F1"] * 100,
            "epochs":   int(r["Epochs_Run"]),
        })
    return rows, len(df)

DISTRIBUTION = [
    # class, train, val, test, total, pct
    ("Glioma",     1534, 329, 328, 2191, 36.9),
    ("Meningioma", 1035, 222, 221, 1478, 24.9),
    ("Pituitary",  1040, 223, 222, 1485, 25.0),
    ("No tumor",    551, 118, 118,  787, 13.2),
]

def analysis_context():
    """Grounded analysis notes pulled from the actual benchmark numbers."""
    rows, n = benchmark_data()
    d = {r["arch"]: r for r in rows}
    notes = []
    # fine-tuned vs frozen gap (real CSV deltas)
    if "DenseNet121_finetuned" in d and "DenseNet121_frozen" in d:
        gap = d["DenseNet121_finetuned"]["accuracy"] - d["DenseNet121_frozen"]["accuracy"]
        notes.append(
            f"Fine-tuning beats frozen features: DenseNet121 goes from "
            f"{d['DenseNet121_frozen']['accuracy']:.1f}% to "
            f"{d['DenseNet121_finetuned']['accuracy']:.1f}% (+\u2212{gap:.1f}pp). "
            f"ImageNet features left frozen are built for natural photos, not "
            f"MRI textures - letting the weights adapt to the scan domain is "
            f"what closes that gap.")
    if "ResNet50_finetuned" in d and "ResNet50_frozen" in d:
        gap = d["ResNet50_finetuned"]["accuracy"] - d["ResNet50_frozen"]["accuracy"]
        notes.append(
            f"The same pattern holds for ResNet50: {d['ResNet50_frozen']['accuracy']:.1f}% "
            f"frozen vs {d['ResNet50_finetuned']['accuracy']:.1f}% fine-tuned "
            f"(+{gap:.1f}pp).")
    # from-scratch story
    if "CNN_scratch" in d:
        notes.append(
            f"From-scratch models still hold up: CNN_scratch reaches "
            f"{d['CNN_scratch']['accuracy']:.1f}% on a clean, leakage-free test "
            f"set - the zero-overlap split means these numbers reflect real "
            f"generalization, not memorized near-duplicates.")
    # transformers note
    if "ViT_scratch" in d and "Hybrid_CNN_ViT_scratch" in d:
        notes.append(
            f"Transformer variants trail the CNNs ({d['ViT_scratch']['accuracy']:.1f}% "
            f"ViT, {d['Hybrid_CNN_ViT_scratch']['accuracy']:.1f}% hybrid) "
            f"- they typically need far more data than ~5.9k images to shine.")
    return notes, n

# ------------------------------------------------------------------------ routes
@app.route("/", methods=["GET", "POST"])
def home():
    payload = None
    if request.method == "POST":
        try:
            f = request.files.get("image")
            if f is None or f.filename == "":
                raise ValueError("Please choose an image file first.")
            ext = f.filename.rsplit(".", 1)[-1].lower()
            if ext not in ALLOWED_EXTS:
                raise ValueError("Unsupported file type. Use .png/.jpg/.jpeg/.bmp/.tif/.tiff")
            payload = finalize(load_image(f))
        except ValueError as e:
            flash(str(e), "danger")
        except Exception:
            flash("Something went wrong while running the models. Try another image.",
                  "danger")
    else:
        sample = request.args.get("sample")
        if sample:
            path = os.path.join(BASE_DIR, "static", "img", "samples", sample)
            if os.path.exists(path):
                payload = finalize(Image.open(path).convert("RGB"))
            else:
                flash("Sample image not found.", "warning")

    benchmark, n_models = benchmark_data()
    analysis_notes, _ = analysis_context()
    return render_template(
        "index.html",
        payload=payload,
        classes=CLASS_NAMES, colors=CLASS_COLORS,
        samples=sorted(os.listdir(os.path.join(BASE_DIR, "static", "img", "samples"))),
        benchmark=benchmark, n_bench=len(benchmark),
        distribution=DISTRIBUTION,
        analysis_notes=analysis_notes,
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: POST multipart 'image' -> per-model predictions + consensus."""
    try:
        f = request.files.get("image")
        if f is None or f.filename == "":
            return jsonify({"error": "missing 'image' file"}), 400
        pil_img = load_image(f)
        (results, con_label, con_conf,
         ens_label, ens_conf, avg, votes) = predict_all(pil_img)
        w = weighted_scoring(results)
        return jsonify({
            "class_names": CLASS_NAMES,
            "models": results,
            "majority_vote": {"label": con_label, "confidence": con_conf},
            "ensemble": {"label": ens_label, "confidence": ens_conf,
                         "probabilities": {c: float(p)
                                           for c, p in zip(CLASS_NAMES, avg)}},
            "weighted": {
                "label": w["label"], "confidence": w["confidence"],
                "scores": w["scores"],
                "prob_score": w["prob_score"],
                "vote_score": w["vote_score"],
                "weights": w["weights"],
                "blend": list(w["blend"]),
            },
            "votes": dict(votes),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "prediction failed"}), 500

@app.errorhandler(413)
def too_large(e):
    flash("File is too large (max 16 MB).", "danger")
    return redirect(url_for("home"))

def _warmup():
    """Load every model and run one inference at startup (background thread)
    so the first real upload doesn't pay the ~1 min cold-start cost."""
    try:
        sample_path = os.path.join(BASE_DIR, "static", "img", "samples",
                                   "sample_no_tumor.jpg")
        if os.path.exists(sample_path):
            predict_all(Image.open(sample_path).convert("RGB"))
    except Exception as e:
        print(f"[warmup] failed: {e}")

threading.Thread(target=_warmup, daemon=True, name="model-warmup").start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)