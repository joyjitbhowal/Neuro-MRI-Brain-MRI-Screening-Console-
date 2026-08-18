"""
BDNeuro-MRI — 4-Class Brain Tumor Classification, Full Model Loop for Google Colab
=====================================================================================
Reproduces the 7-model benchmark table from the reference dataset page
"BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset for Four-Class
Deep Learning Classification" (Mendeley Data, DOI 10.17632/zwr4ntf94j.7).

Dataset facts (verified by this script at runtime):
  5,941 T1-weighted contrast-enhanced MRI images
  Classes:  glioma, meningioma, pituitary, no_tumor
  Split:    train=4,160 (70%)  val=892 (15%)  test=889 (15%), zero leakage

BEFORE RUNNING:
1. Runtime > Change runtime type > GPU (T4 or better). ViT/Hybrid from scratch
   are very slow on CPU.
2. Upload `Brain_Tumor_MRI_Dataset_Final.zip` (107 MB, from Mendeley) to your
   Google Drive root — or change ZIP_PATH below to its exact Drive path/filename.

CELLS:
  CELL 1-4  setup + model builders      (run once)
  CELL 5    train loop (RESUMABLE — skips models already trained)
  CELL 6    FINAL comparison suite — every graph, metric, chart + web-app
            package zip (run after ALL 7 models are trained)
  CELL 7    OPTIONAL live UI: upload one MRI, compare ALL models (Gradio)

Notes:
  * Uses tf.keras.utils.image_dataset_from_directory (Keras 3 compatible).
    ImageDataGenerator is REMOVED in current Colab TF, so it is NOT used here.
  * All outputs (models, benchmark CSV, charts, assets) are saved to
    DRIVE_OUTPUT_DIR so nothing is lost on a runtime disconnect.
"""

# %% CELL 1 — Mount Drive, find & unzip the dataset
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile

ZIP_PATH        = "/content/drive/MyDrive/Brain_Tumor_MRI_Dataset_Final.zip"  # <-- exact Drive path/filename
EXTRACT_DIR     = "/content/dataset"                     # local unzip = fast I/O during training
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/brain_tumor_project"  # <-- everything saved here

os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)

def safe_extract(zip_path, target_dir):
    """Extract a zip while blocking path-traversal ('zip slip') entries —
    refuses to write any file that would land outside target_dir."""
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Zip not found at: {zip_path}\n"
            f"List your Drive with:  os.listdir('/content/drive/MyDrive')\n"
            f"and set ZIP_PATH to the exact name (mine should be a .zip file)."
        )
    target_dir = os.path.realpath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.infolist():
            member_path = os.path.realpath(os.path.join(target_dir, member.filename))
            if not (member_path == target_dir or member_path.startswith(target_dir + os.sep)):
                raise ValueError(f"Blocked unsafe path in zip: {member.filename}")
        z.extractall(target_dir)  # safe now — every member validated above

if not os.path.exists(EXTRACT_DIR):
    safe_extract(ZIP_PATH, EXTRACT_DIR)

# Find the folder that actually contains train/val/test (handles nested zip roots)
DATA_DIR = EXTRACT_DIR
for root, dirs, files in os.walk(EXTRACT_DIR):
    if {'train', 'val', 'test'}.issubset(set(dirs)):
        DATA_DIR = root
        break

print("Using DATA_DIR:", DATA_DIR)
print("Contents:", sorted(os.listdir(DATA_DIR)))

# ---- Sanity check: counts must match the dataset's published split ----
import collections
for split in ['train', 'val', 'test']:
    counts = collections.Counter()
    for cls in sorted(os.listdir(f"{DATA_DIR}/{split}")):
        d = f"{DATA_DIR}/{split}/{cls}"
        if os.path.isdir(d):
            counts[cls] = len([f for f in os.listdir(d) if os.path.isfile(f"{d}/{f}")])
    total = sum(counts.values())
    print(f"{split:5s} total={total:<5d} " + "  ".join(f"{c}={n}" for c, n in counts.items()))
print("(Reference: train=4160, val=892, test=889, total=5941)")


# %% CELL 2 — Imports & config
import time, json, gc, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.metrics import f1_score, classification_report

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

IMG_SIZE    = (224, 224)   # drop to (160, 160) if you're short on time
BATCH_SIZE  = 32
EPOCHS      = 15           # early stopping usually cuts this short
PATIENCE    = 4
NUM_CLASSES = 4
MODEL_DIR   = f"{DRIVE_OUTPUT_DIR}/models"
os.makedirs(MODEL_DIR, exist_ok=True)


# %% CELL 3 — Model builders
def build_cnn():
    return models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'), layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

def _transformer_block(x, d_model, num_heads, mlp_dim):
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x1, x1)
    x2 = layers.Add()([attn, x])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    mlp = layers.Dense(mlp_dim, activation='gelu')(x3)
    mlp = layers.Dense(d_model)(mlp)
    return layers.Add()([mlp, x2])

def build_vit_scratch(patch_size=16, num_layers=4, d_model=128, num_heads=4, mlp_dim=256):
    img_size = IMG_SIZE[0]
    num_patches = (img_size // patch_size) ** 2
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Conv2D(d_model, kernel_size=patch_size, strides=patch_size)(inputs)
    x = layers.Reshape((num_patches, d_model))(x)
    pos = layers.Embedding(input_dim=num_patches, output_dim=d_model)(tf.range(num_patches))
    x = x + pos
    for _ in range(num_layers):
        x = _transformer_block(x, d_model, num_heads, mlp_dim)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return models.Model(inputs, outputs)

def build_hybrid_cnn_vit():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)          # 112
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)          # 56
    x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)  # 28
    shape = x.shape
    num_patches, d_model = shape[1] * shape[2], shape[3]
    x = layers.Reshape((num_patches, d_model))(x)
    pos = layers.Embedding(input_dim=num_patches, output_dim=d_model)(tf.range(num_patches))
    x = x + pos
    for _ in range(2):
        x = _transformer_block(x, d_model, num_heads=4, mlp_dim=256)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return models.Model(inputs, outputs)

def build_pretrained(base_arch, fine_tune=False):
    if base_arch == 'resnet50':
        base = applications.ResNet50(include_top=False, weights='imagenet',
                                      input_shape=(*IMG_SIZE, 3), pooling='avg')
    elif base_arch == 'densenet121':
        base = applications.DenseNet121(include_top=False, weights='imagenet',
                                         input_shape=(*IMG_SIZE, 3), pooling='avg')
    base.trainable = fine_tune
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs)   # training flag left default: correct BN stats for train & eval
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return models.Model(inputs, outputs)


# %% CELL 4 — The loop config: name -> (builder, preprocessing fn)
MODEL_CONFIGS = {
    "CNN_scratch":            (build_cnn,                                     lambda x: tf.cast(x, tf.float32) / 255.0),
    "ViT_scratch":            (build_vit_scratch,                             lambda x: tf.cast(x, tf.float32) / 255.0),
    "Hybrid_CNN_ViT_scratch": (build_hybrid_cnn_vit,                          lambda x: tf.cast(x, tf.float32) / 255.0),
    "ResNet50_frozen":        (lambda: build_pretrained('resnet50', False),   applications.resnet50.preprocess_input),
    "DenseNet121_frozen":     (lambda: build_pretrained('densenet121', False), applications.densenet.preprocess_input),
    "ResNet50_finetuned":     (lambda: build_pretrained('resnet50', True),    applications.resnet50.preprocess_input),
    "DenseNet121_finetuned":  (lambda: build_pretrained('densenet121', True),  applications.densenet.preprocess_input),
}


# %% CELL 5 — Run the full loop (RESUMABLE: skips already-trained models)
AUTOTUNE = tf.data.AUTOTUNE

results_file = f"{DRIVE_OUTPUT_DIR}/model_benchmark_results.csv"
results = {}                                    # name -> result row
if os.path.exists(results_file):
    for _, row in pd.read_csv(results_file).iterrows():
        results[row['Architecture']] = row.to_dict()
    print(f"Resumed {len(results)} recorded result(s): "
          + ", ".join(results.keys()) or "(none)")

for name, (builder, preprocess_fn) in MODEL_CONFIGS.items():
    weights_path = f"{MODEL_DIR}/{name}.keras"

    # --- Skip / recover models that are already trained ---
    if os.path.exists(weights_path):
        if name in results:
            print(f"\n{name}: ALREADY TRAINED -> skipping.")
            continue
        print(f"\n{name}: weights found but no results recorded -> evaluating saved model.")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            f"{DATA_DIR}/test",  image_size=IMG_SIZE, batch_size=BATCH_SIZE,
            label_mode='categorical', shuffle=False)
        def _p(x, y):
            return preprocess_fn(x), y
        test_ds = test_ds.map(_p, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
        model = tf.keras.models.load_model(weights_path)
        y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
        results[name] = {
            "Architecture": name,
            "Test_Accuracy": round(float(np.mean(y_pred == y_true)), 4),
            "Test_F1": round(float(f1_score(y_true, y_pred, average='weighted')), 4),
            "Epochs_Run": "best",
            "Train_Time_sec": "n/a",
        }
        pd.DataFrame(list(results.values())).to_csv(results_file, index=False)
        continue

    # --- Train the remaining models ---
    print(f"\n{'='*60}\nTraining: {name}\n{'='*60}")

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=True, seed=SEED)

    def _aug_then_preprocess(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.05)
        x = tf.clip_by_value(x, 0.0, 255.0)
        return preprocess_fn(x), y

    train_ds = (train_ds_raw.map(_aug_then_preprocess, num_parallel_calls=AUTOTUNE)
                            .prefetch(AUTOTUNE))

    val_ds_raw  = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/val",  image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=False)
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='categorical', shuffle=False)

    class_names = test_ds_raw.class_names

    def _preprocess(x, y):
        return preprocess_fn(x), y

    val_ds  = val_ds_raw.map(_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds_raw.map(_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])

    model = builder()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(weights_path, save_best_only=True, monitor='val_accuracy'),
    ]

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                        callbacks=callbacks, verbose=1)
    train_time = time.time() - start

    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    results[name] = {
        "Architecture": name,
        "Test_Accuracy": round(float(np.mean(y_pred == y_true)), 4),
        "Test_F1": round(float(f1_score(y_true, y_pred, average='weighted')), 4),
        "Epochs_Run": len(history.history['loss']),
        "Train_Time_sec": round(train_time, 1),
    }

    pd.DataFrame(list(results.values())).to_csv(results_file, index=False)

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(f"{MODEL_DIR}/{name}_classification_report.csv")

    print(f"-> {name}: Acc={results[name]['Test_Accuracy']}  "
          f"F1={results[name]['Test_F1']}  ({train_time:.0f}s)")

    del model
    gc.collect()
    tf.keras.backend.clear_session()

results_df = pd.DataFrame(list(results.values())).sort_values("Test_Accuracy", ascending=False)
results_df.to_csv(results_file, index=False)
print("\nFINAL RESULTS")
print(results_df.to_string(index=False))


# %% CELL 6 — FINAL comparison suite (run AFTER all 7 models are trained)
import matplotlib.pyplot as plt
import shutil as _shutil
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

AUTOTUNE   = tf.data.AUTOTUNE
WEB_ASSETS = f"{DRIVE_OUTPUT_DIR}/web_app_assets"
os.makedirs(f"{WEB_ASSETS}/confusion", exist_ok=True)
plt.rcParams.update({"figure.max_open_warning": 60})

PREPROCESS_MODE = {
    "CNN_scratch":            "rescale_255",
    "ViT_scratch":            "rescale_255",
    "Hybrid_CNN_ViT_scratch": "rescale_255",
    "ResNet50_frozen":        "resnet50",
    "DenseNet121_frozen":     "densenet121",
    "ResNet50_finetuned":     "resnet50",
    "DenseNet121_finetuned":  "densenet121",
}

class_names = sorted(d for d in os.listdir(f"{DATA_DIR}/test")
                     if os.path.isdir(f"{DATA_DIR}/test/{d}"))

test_raw = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode='categorical', shuffle=False)
y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_raw])
N = len(y_true)

def _pp_for(mode):
    if mode == "rescale_255":
        return lambda x: tf.cast(x, tf.float32) / 255.0
    if mode == "resnet50":
        return applications.resnet50.preprocess_input
    return applications.densenet.preprocess_input

# ---------- evaluate every trained model ----------
rows, conf_cmats, miss = [], {}, []
for name, mode in PREPROCESS_MODE.items():
    weights_path = f"{MODEL_DIR}/{name}.keras"
    if not os.path.exists(weights_path):
        miss.append(name)
        print(f"[skip] {name}: weights not found — train it first (Cell 5).")
        continue

    model = tf.keras.models.load_model(weights_path)
    n_params      = int(model.count_params())
    n_train       = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    pp            = _pp_for(mode)
    ds            = test_raw.map(lambda x, y: (pp(x), y),
                                 num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    model.predict(ds.take(1), verbose=0)                 # warm-up
    t0 = time.perf_counter()
    probs = model.predict(ds, verbose=0)
    lat_ms = (time.perf_counter() - t0) / N * 1000.0     # ms per image
    y_pred = np.argmax(probs, axis=1)
    del model
    tf.keras.backend.clear_session()

    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    conf_cmats[name] = cm
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        f"{WEB_ASSETS}/confusion/{name}_confusion.csv")
    cm_norm = cm.astype("float") / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    pd.DataFrame(np.round(cm_norm, 4), index=class_names, columns=class_names).to_csv(
        f"{WEB_ASSETS}/confusion/{name}_confusion_normalized.csv")

    rep = classification_report(y_true, y_pred, target_names=class_names,
                                output_dict=True, zero_division=0)
    row = {
        "Architecture": name,
        "Test_Accuracy":      round(accuracy_score(y_true, y_pred), 4),
        "Weighted_F1":        round(f1_score(y_true, y_pred, average='weighted'), 4),
        "Macro_F1":           round(f1_score(y_true, y_pred, average='macro'), 4),
        "Macro_Precision":    round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Macro_Recall":       round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "Parameters_M":       round(n_params / 1e6, 2),
        "Trainable_M":        round(n_train / 1e6, 2),
        "Latency_ms/img":     round(lat_ms, 2),
    }
    for c in class_names:
        row[f"prec_{c}"] = round(rep[c]['precision'], 4)
        row[f"rec_{c}"]  = round(rep[c]['recall'], 4)
        row[f"f1_{c}"]   = round(rep[c]['f1-score'], 4)
    rows.append(row)
    print(f"{name}: Acc={row['Test_Accuracy']}  F1={row['Weighted_F1']}  "
          f"lat={row['Latency_ms/img']}ms  params={row['Parameters_M']}M")

if miss:
    print(f"\nWARNING: not trained yet -> {miss}. Results/charts below cover the rest.")

if not rows:
    raise SystemExit("No trained models found in " + MODEL_DIR
                     + " — run Cell 5 first, then re-run this cell.")

full = pd.DataFrame(rows).sort_values(["Test_Accuracy", "Weighted_F1"], ascending=False)
full.insert(0, "Rank", range(1, len(full) + 1))
full.to_csv(f"{WEB_ASSETS}/per_model_metrics.csv", index=False)
print("\n" + full[["Rank", "Architecture", "Test_Accuracy", "Weighted_F1",
                   "Macro_F1", "Latency_ms/img", "Parameters_M"]].to_string(index=False))

# ---------- G1: Accuracy & Weighted-F1 grouped bar ----------
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(full)); w = 0.4
ax.bar(x - w/2, full["Test_Accuracy"], w, label="Accuracy",    color="#1f77b4")
ax.bar(x + w/2, full["Weighted_F1"],   w, label="Weighted F1", color="#ff7f0e")
for xi, acc in zip(x - w/2, full["Test_Accuracy"]):
    ax.text(xi, acc + 0.008, f"{acc:.3f}", ha="center", fontsize=8)
for xi, fi in zip(x + w/2, full["Weighted_F1"]):
    ax.text(xi, fi + 0.008, f"{fi:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(full["Architecture"], rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Test Accuracy & Weighted F1 by Model")
plt.tight_layout(); plt.savefig(f"{WEB_ASSETS}/overall_accuracy_f1.png", dpi=150)
plt.show(); plt.close(fig)

# ---------- G2: Macro metrics grouped bar ----------
fig, ax = plt.subplots(figsize=(11, 5))
w = 0.25
for j, col, c in [(0, "Macro_F1", "#2ca02c"), (1, "Macro_Precision", "#9467bd"),
                  (2, "Macro_Recall", "#d62728")]:
    ax.bar(x + (j - 1) * w, full[col], w, label=col.replace("Macro_", "Macro "), color=c)
ax.set_xticks(x); ax.set_xticklabels(full["Architecture"], rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Macro F1 / Precision / Recall by Model")
plt.tight_layout(); plt.savefig(f"{WEB_ASSETS}/macro_metrics.png", dpi=150)
plt.show(); plt.close(fig)

# ---------- G3/G4: per-class recall & precision grouped bars ----------
rec_cols  = [f"rec_{c}"  for c in class_names]
prec_cols = [f"prec_{c}" for c in class_names]
nw, bw = len(rec_cols), 0.8 / len(rec_cols)

fig, ax = plt.subplots(figsize=(12, 6))
for j, col in enumerate(rec_cols):
    ax.bar(x + (j - nw/2 + 0.5) * bw, full[col], bw, label=col.replace("rec_", ""))
ax.set_xticks(x); ax.set_xticklabels(full["Architecture"], rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.legend(title="Class recall", ncol=nw, fontsize=8)
ax.set_ylabel("Recall"); ax.set_title("Per-Class Recall by Model")
plt.tight_layout(); plt.savefig(f"{WEB_ASSETS}/per_class_recall.png", dpi=150)
plt.show(); plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 6))
for j, col in enumerate(prec_cols):
    ax.bar(x + (j - nw/2 + 0.5) * bw, full[col], bw, label=col.replace("prec_", ""))
ax.set_xticks(x); ax.set_xticklabels(full["Architecture"], rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.legend(title="Class precision", ncol=nw, fontsize=8)
ax.set_ylabel("Precision"); ax.set_title("Per-Class Precision by Model")
plt.tight_layout(); plt.savefig(f"{WEB_ASSETS}/per_class_precision.png", dpi=150)
plt.show(); plt.close(fig)

# ---------- G5/G6: model x class heatmaps ----------
for metric, cols, cmap, fname in [("Recall", rec_cols, "Blues", "recall_heatmap.png"),
                                  ("Precision", prec_cols, "Greens", "precision_heatmap.png")]:
    fig, ax = plt.subplots(figsize=(11, max(4, len(full) * 0.6)))
    data = full[cols].values
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(full))); ax.set_yticklabels(full["Architecture"], fontsize=9)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=30, ha="right")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im); ax.set_title(f"Model vs Class — {metric}")
    plt.tight_layout(); plt.savefig(f"{WEB_ASSETS}/{fname}", dpi=150)
    plt.show(); plt.close(fig)

# ---------- G7: confusion matrices (absolute + normalized) per model ----------
for name, cm in conf_cmats.items():
    for cm_, tag in [(cm, ""),
                     (np.round(cm / np.clip(cm.sum(1, keepdims=True), 1, None), 3), "_normalized")]:
        disp = ConfusionMatrixDisplay(cm_, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(5.6, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False,
                  values_format=".2f" if tag else "d")
        ax.set_title(f"{name}{tag.replace('_', ' ')}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{WEB_ASSETS}/confusion/{name}_confusion{tag}.png", dpi=120)
        plt.close(fig)

# ---------- G8: 2x2 dashboard (one big comparison figure) ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax = axes[0, 0]; w = 0.4
ax.bar(x - w/2, full["Test_Accuracy"], w, label="Accuracy",    color="#1f77b4")
ax.bar(x + w/2, full["Weighted_F1"],   w, label="Weighted F1", color="#ff7f0e")
ax.set_xticks(x); ax.set_xticklabels(full["Architecture"], rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.set_title("Accuracy & Weighted F1")

ax = axes[0, 1]
ax.bar(x - w/2, full["Macro_F1"],     w, label="Macro F1", color="#2ca02c")
ax.bar(x + w/2, full["Macro_Recall"], w, label="Macro Recall", color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(full["Architecture"], rotation=30, ha="right")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.set_title("Macro F1 & Recall")

ax = axes[1, 0]
data = full[rec_cols].values
im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_yticks(range(len(full))); ax.set_yticklabels(full["Architecture"], fontsize=9)
ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=30, ha="right")
plt.colorbar(im, ax=ax, fraction=0.046); ax.set_title("Class Recall")

ax = axes[1, 1]
ax.bar(x, full["Latency_ms/img"], color="#8c564b")
ax.set_xticks(x); ax.set_xticklabels(full["Architecture"], rotation=30, ha="right")
for xi, lat in zip(x, full["Latency_ms/img"]):
    ax.text(xi, lat + max(full["Latency_ms/img"]) * 0.02, f"{lat:.0f}ms",
            ha="center", fontsize=8)
ax.set_ylabel("ms / image"); ax.set_title("Inference Speed (lower is better)")

fig.suptitle("BDNeuro-MRI — 7-Model Comparison Dashboard", fontsize=15)
plt.tight_layout(); plt.savefig(f"{WEB_ASSETS}/comparison_dashboard.png", dpi=150)
plt.show(); plt.close(fig)

# ---------- G9: speed vs accuracy trade-off ----------
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(full["Latency_ms/img"], full["Test_Accuracy"],
                s=full["Parameters_M"].values * 8 + 80,
                c=x, cmap="viridis", alpha=0.8)
for _, r in full.iterrows():
    ax.annotate(r["Architecture"], (r["Latency_ms/img"], r["Test_Accuracy"]),
                textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("Inference time (ms/image)  →  faster is to the LEFT")
ax.set_ylabel("Test Accuracy  →  better is UP")
ax.set_title("Accuracy-Speed Trade-off (bubble size = parameters)")
plt.colorbar(sc, label="")
plt.tight_layout(); plt.savefig(f"{WEB_ASSETS}/speed_accuracy_tradeoff.png", dpi=150)
plt.show(); plt.close(fig)

# ---------- manifest for the web app / live UI ----------
manifest = {
    "dataset": "BDNeuro-MRI (Mendeley DOI 10.17632/zwr4ntf94j.7)",
    "img_size": list(IMG_SIZE),
    "class_names": class_names,
    "test_samples": int(N),
    "models_dir": "models/",
    "models": {n: {"file": f"models/{n}.keras", "preprocessing": m}
               for n, m in PREPROCESS_MODE.items()},
    "metrics_csv": "web_app_assets/per_model_metrics.csv",
    "charts": ["overall_accuracy_f1.png", "macro_metrics.png",
               "per_class_recall.png", "per_class_precision.png",
               "recall_heatmap.png", "precision_heatmap.png",
               "comparison_dashboard.png", "speed_accuracy_tradeoff.png"],
    "confusion_dir": "web_app_assets/confusion/",
}
with open(f"{WEB_ASSETS}/model_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# ---------- one ready-to-download package for the website ----------
pkg = _shutil.make_archive(f"{DRIVE_OUTPUT_DIR}/brain_tumor_webapp_package",
                           "zip", root_dir=DRIVE_OUTPUT_DIR)
print(f"\nWeb-app package (models + assets + charts): {pkg}")

print("\nBEST MODEL: " + full.iloc[0]["Architecture"]
      + f" (Acc={full.iloc[0]['Test_Accuracy']}, "
      + f"WeightedF1={full.iloc[0]['Weighted_F1']})")

print("\nALL SAVED under:", DRIVE_OUTPUT_DIR)
for f_ in sorted(os.listdir(WEB_ASSETS)) + ["confusion/ (abs + normalized PNG/CSV per model)"]:
    print("   web_app_assets/", f_)


# %% CELL 7 — OPTIONAL live UI: upload one MRI, compare ALL trained models side-by-side
# First run: uncomment the line below to install gradio (takes ~30s)
# !pip install -q gradio

import gradio as gr
from PIL import Image

with open(f"{WEB_ASSETS}/model_manifest.json") as f:
    MAN = json.load(f)
CLASS_NAMES = MAN["class_names"]
IMG_W, IMG_H = MAN["img_size"][0], MAN["img_size"][1]

_MODELS = {}
def _load(name):
    if name not in _MODELS:
        _MODELS[name] = tf.keras.models.load_model(f"{MODEL_DIR}/{name}.keras")
    return _MODELS[name]

def _prep(img, mode):
    x = np.asarray(img, dtype="float32")          # RGB, values 0-255
    if mode == "rescale_255":
        x = x / 255.0
    elif mode == "resnet50":
        x = applications.resnet50.preprocess_input(x)
    else:  # densenet121
        x = applications.densenet.preprocess_input(x)
    return x[None, ...]                           # add batch dim

def compare_models(img):
    img = img.convert("RGB").resize((IMG_W, IMG_H))
    rows = []
    for name, info in MAN["models"].items():
        if not os.path.exists(f"{MODEL_DIR}/{name}.keras"):
            continue
        model = _load(name)
        probs = model.predict(_prep(img, info["preprocessing"]), verbose=0)[0]
        top = int(np.argmax(probs))
        rows.append((name, CLASS_NAMES[top], float(probs[top]), probs))

    html = ("<h3>Multimodel comparison</h3>"
            "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
            "<tr><th>Model</th><th>Predicted class</th><th>Confidence</th></tr>")
    for name, label, conf, _ in rows:
        html += f"<tr><td>{name}</td><td>{label}</td><td>{conf:.1%}</td></tr>"
    html += "</table>"

    fig, ax = plt.subplots(figsize=(9, len(rows) * 0.8 + 1))
    for i, (name, label, conf, probs) in enumerate(rows):
        ax.barh(i, conf, color="#2a9d8f")
        ax.text(conf + 0.01, i, f"{label} ({conf:.1%})", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Confidence in predicted class")
    ax.set_title("Each model's top prediction")
    plt.tight_layout()
    return gr.HTML(html), fig

gr.Interface(
    fn=compare_models,
    inputs=gr.Image(type="pil", label="Upload a brain MRI"),
    outputs=[gr.HTML(label="Predictions"), gr.Plot(label="Confidence")],
    title="BDNeuro-MRI — 7-Model Comparison",
    description="Upload a brain MRI image. Every trained model predicts "
                "independently and results are shown side-by-side.",
).launch(debug=True, share=False)   # share=False -> runs in Colab; share=True -> public URL