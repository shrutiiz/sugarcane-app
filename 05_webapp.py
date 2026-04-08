"""
CODE 05 — Web Application (Flask — Permanent Deployment)
=========================================================
This version automatically downloads artifacts.zip from Google Drive
and extracts it if the local artifacts/ folder is missing.
"""

import os
import json
import base64
import tempfile
import io
import zipfile
import shutil
import traceback
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import threading
PREDICT_LOCK = threading.Lock()

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.getcwd()
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
ARTIFACT_ZIP_PATH = os.path.join(BASE_DIR, "artifacts.zip")

# Paste your Google Drive ZIP file ID here
ARTIFACTS_ZIP_FILE_ID = "1132g9o2Wk1zr6zWf7yt5nd7wzCq5Ld_b"

CNN_WEIGHTS_PATH         = os.path.join(ARTIFACT_DIR, "best_efficientnet_model.weights.h5")
CLASS_NAMES_PATH         = os.path.join(ARTIFACT_DIR, "image_class_names.json")
TEMP_SCALE_PATH          = os.path.join(ARTIFACT_DIR, "temperature_scale.json")
ENV_DISEASE_MODEL_PATH   = os.path.join(ARTIFACT_DIR, "env_disease_prior_model.pkl")
ENV_DISEASE_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "env_disease_label_encoder.pkl")
REC_MODEL_PATH           = os.path.join(ARTIFACT_DIR, "fertilizer_recommendation_model.pkl")
FERT_ENCODER_PATH        = os.path.join(ARTIFACT_DIR, "fertilizer_label_encoder.pkl")
SOIL_ENCODER_PATH        = os.path.join(ARTIFACT_DIR, "soil_encoder.pkl")
CROP_ENCODER_PATH        = os.path.join(ARTIFACT_DIR, "crop_encoder.pkl")
NUMERIC_SCALER_PATH      = os.path.join(ARTIFACT_DIR, "numeric_scaler.pkl")

IMG_SIZE = (128, 128)

CANONICAL_DISEASE_CLASSES = [
    "BacterialBlight", "Healthy", "Mosaic", "RedRot", "Rust", "YellowLeaf"
]

CLASS_NAME_MAP = {
    "BacterialBlight":     "BacterialBlight",
    "BacterialBlights":    "BacterialBlight",
    "Healthy":             "Healthy",
    "Mosaic":              "Mosaic",
    "RedRot":              "RedRot",
    "Rust":                "Rust",
    "Yellow":              "YellowLeaf",
    "YellowLeaf":          "YellowLeaf",
    "Yellow Leaf Disease": "YellowLeaf",
    "YellowLeafDisease":   "YellowLeaf"
}

NUMERIC_COLS = [
    "Temparature", "Humidity", "Moisture",
    "Nitrogen", "Potassium", "Phosphorous"
]

ENV_FEATURE_COLS = NUMERIC_COLS + ["Soil Type Enc", "Crop Type Enc"]

REC_FEATURE_COLS = ENV_FEATURE_COLS + [
    "dprob_BacterialBlight", "dprob_Healthy", "dprob_Mosaic",
    "dprob_RedRot", "dprob_Rust", "dprob_YellowLeaf"
]

PESTICIDE_ADVISORY = {
    "BacterialBlight": [
        "Apply copper-based bactericide (e.g. Blitox 50 WP) — 3g/L water",
        "Improve field drainage to reduce moisture buildup",
        "Remove and destroy severely infected cane stalks"
    ],
    "Healthy": [
        "No pesticide treatment needed",
        "Continue balanced NPK fertilization schedule",
        "Monitor for early symptom signs every 2 weeks"
    ],
    "Mosaic": [
        "Control aphid vectors using systemic insecticide (e.g. Imidacloprid)",
        "Remove and burn severely mosaic-infected plants",
        "Use certified disease-free seed cane for next crop"
    ],
    "RedRot": [
        "Apply Mancozeb or Carbendazim fungicide as soil drench",
        "Avoid waterlogging — install proper drainage channels",
        "Treat seed cane with Bavistin before planting"
    ],
    "Rust": [
        "Apply Propiconazole or Mancozeb fungicide spray",
        "Monitor fields regularly during high humidity periods",
        "Reduce plant density to improve air circulation"
    ],
    "YellowLeaf": [
        "Apply balanced fertilizer focusing on Potassium and Micronutrients",
        "Use foliar spray of 0.5% ZnSO4 + 0.5% MgSO4",
        "Monitor for whitefly and aphid vectors of YLD virus"
    ]
}

MODELS = {}

# ============================================================
# DOWNLOAD + EXTRACT ARTIFACTS
# ============================================================

def download_file_from_drive(file_id, destination):
    import gdown
    import os

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False, fuzzy=True)

    print("[DEBUG] Saved:", destination)
    print("[DEBUG] Size:", os.path.getsize(destination))

def ensure_artifacts_downloaded():
    if os.path.exists(CNN_WEIGHTS_PATH):
        return

    if not ARTIFACTS_ZIP_FILE_ID or ARTIFACTS_ZIP_FILE_ID == "PASTE_YOUR_GOOGLE_DRIVE_ZIP_FILE_ID_HERE":
        raise RuntimeError("Set ARTIFACTS_ZIP_FILE_ID in 05_webapp.py")

    print("[INFO] artifacts/ not found. Downloading artifacts.zip from Google Drive...")

    if os.path.exists(ARTIFACT_DIR):
        shutil.rmtree(ARTIFACT_DIR)

    if os.path.exists(ARTIFACT_ZIP_PATH):
        os.remove(ARTIFACT_ZIP_PATH)

    download_file_from_drive(ARTIFACTS_ZIP_FILE_ID, ARTIFACT_ZIP_PATH)

    print("[DEBUG] Downloaded file path:", ARTIFACT_ZIP_PATH)
    print("[DEBUG] Downloaded file size:", os.path.getsize(ARTIFACT_ZIP_PATH))

    with zipfile.ZipFile(ARTIFACT_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)

    print("[INFO] Artifacts extracted successfully.")
# ============================================================
# HELPERS
# ============================================================

def check_file_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")

def normalize_prob_dict(prob_dict):
    keys = list(prob_dict.keys())
    vals = np.array(list(prob_dict.values()), dtype=np.float32)
    vals = np.clip(vals, 1e-9, None)
    vals = vals / vals.sum()
    return {k: float(v) for k, v in zip(keys, vals)}

def to_canonical_name(name):
    return CLASS_NAME_MAP.get(str(name).strip(), str(name).strip())

def apply_temperature_scaling(raw_probs, T):
    logits = np.log(raw_probs + 1e-9)
    scaled = logits / T
    shifted = scaled - scaled.max()
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum()

def safe_label_transform(encoder, value, field_name):
    value = str(value).strip()
    if value not in encoder.classes_:
        raise ValueError(f"Unknown {field_name}: '{value}'. Allowed: {list(encoder.classes_)}")
    return int(encoder.transform([value])[0])

def to_python(obj):
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(x) for x in obj]
    elif isinstance(obj, tuple):
        return [to_python(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ============================================================
# REBUILD CNN MODEL — must match Code 00 exactly
# ============================================================

def build_cnn_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    attn = tf.keras.layers.Dense(1280 // 4, activation="relu", name="attn_squeeze")(x)
    attn = tf.keras.layers.Dense(1280, activation="sigmoid", name="attn_excite")(attn)
    x = tf.keras.layers.Multiply(name="attn_weighted")([x, attn])

    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return tf.keras.Model(inputs, outputs, name="EfficientNet_AttentionHead")

# ============================================================
# LOAD ALL MODELS ONCE
# ============================================================

def load_all_models():
    print("[STARTUP] Loading all models...")

    ensure_artifacts_downloaded()

    required = {
        "CNN weights": CNN_WEIGHTS_PATH,
        "Class names": CLASS_NAMES_PATH,
        "Env disease model": ENV_DISEASE_MODEL_PATH,
        "Env disease encoder": ENV_DISEASE_ENCODER_PATH,
        "Recommendation model": REC_MODEL_PATH,
        "Fertilizer encoder": FERT_ENCODER_PATH,
        "Soil encoder": SOIL_ENCODER_PATH,
        "Crop encoder": CROP_ENCODER_PATH,
        "Numeric scaler": NUMERIC_SCALER_PATH
    }
    for name, path in required.items():
        check_file_exists(path, name)

    with open(CLASS_NAMES_PATH) as f:
        MODELS["class_names"] = json.load(f)

    MODELS["cnn"] = build_cnn_model(num_classes=len(MODELS["class_names"]))
    MODELS["cnn"].load_weights(CNN_WEIGHTS_PATH)

    MODELS["temperature"] = 1.0
    if os.path.exists(TEMP_SCALE_PATH):
        with open(TEMP_SCALE_PATH) as f:
            MODELS["temperature"] = json.load(f).get("temperature", 1.0)

    MODELS["env_model"]    = joblib.load(ENV_DISEASE_MODEL_PATH)
    MODELS["env_encoder"]  = joblib.load(ENV_DISEASE_ENCODER_PATH)
    MODELS["rec_model"]    = joblib.load(REC_MODEL_PATH)
    MODELS["fert_encoder"] = joblib.load(FERT_ENCODER_PATH)
    MODELS["soil_encoder"] = joblib.load(SOIL_ENCODER_PATH)
    MODELS["crop_encoder"] = joblib.load(CROP_ENCODER_PATH)
    MODELS["scaler"]       = joblib.load(NUMERIC_SCALER_PATH)

    print(f"[STARTUP] Temperature scaling T = {MODELS['temperature']:.4f}")
    print("[STARTUP] All models loaded successfully.")

# ============================================================
# INFERENCE FUNCTIONS
# ============================================================

def predict_image_disease_probs(image_path):
    cnn = MODELS["cnn"]
    class_names = MODELS["class_names"]
    T = MODELS["temperature"]

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_arr = tf.keras.utils.img_to_array(img)
    img_arr = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(img_arr, axis=0)
    )

    raw_preds = cnn.predict(img_arr, verbose=0)[0]
    cal_preds = apply_temperature_scaling(raw_preds, T)

    raw_probs = {cls: float(p) for cls, p in zip(class_names, cal_preds)}

    canonical_probs = {d: 0.0 for d in CANONICAL_DISEASE_CLASSES}
    for raw_name, prob in raw_probs.items():
        canonical = to_canonical_name(raw_name)
        if canonical in canonical_probs:
            canonical_probs[canonical] += prob

    return normalize_prob_dict(canonical_probs)

def preprocess_env_input(env_input):
    df = pd.DataFrame([env_input]).copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df[NUMERIC_COLS] = MODELS["scaler"].transform(df[NUMERIC_COLS])
    df["Soil Type Enc"] = safe_label_transform(MODELS["soil_encoder"], df.loc[0, "Soil Type"], "Soil Type")
    df["Crop Type Enc"] = safe_label_transform(MODELS["crop_encoder"], df.loc[0, "Crop Type"], "Crop Type")
    return df

def predict_env_disease_probs(env_df):
    probs = MODELS["env_model"].predict_proba(env_df[ENV_FEATURE_COLS])[0]
    class_names = MODELS["env_encoder"].inverse_transform(np.arange(len(probs)))
    out = {to_canonical_name(c): float(p) for c, p in zip(class_names, probs)}
    full = {d: out.get(d, 0.0) for d in CANONICAL_DISEASE_CLASSES}
    return normalize_prob_dict(full)

def fuse_disease_probabilities(image_probs, env_probs):
    confidence_img = max(image_probs.values())
    confidence_env = max(env_probs.values())
    alpha = confidence_img / (confidence_img + confidence_env + 1e-8)

    fused = {}
    for d in CANONICAL_DISEASE_CLASSES:
        fused[d] = alpha * image_probs.get(d, 0.0) + (1 - alpha) * env_probs.get(d, 0.0)

    return normalize_prob_dict(fused), float(alpha)

def predict_final_recommendations(env_df, fused_disease_probs):
    fert_encoder = MODELS["fert_encoder"]
    rec_model = MODELS["rec_model"]

    final_scores = np.zeros(len(fert_encoder.classes_), dtype=np.float32)

    for disease_name, disease_prob in fused_disease_probs.items():
        temp_df = env_df.copy()
        for d in CANONICAL_DISEASE_CLASSES:
            temp_df[f"dprob_{d}"] = 0.0
        temp_df[f"dprob_{disease_name}"] = 1.0

        fert_probs = rec_model.predict_proba(temp_df[REC_FEATURE_COLS])[0]
        final_scores += float(disease_prob) * fert_probs

    final_scores = final_scores / final_scores.sum()
    fert_names = fert_encoder.inverse_transform(np.arange(len(final_scores)))

    result = {str(k): float(v) for k, v in zip(fert_names, final_scores)}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam_base64(image_path):
    return None

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app)

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    if request.path.startswith("/predict") or request.path.startswith("/health"):
        response = e.get_response()
        response.data = json.dumps({
            "error": e.name,
            "code": e.code,
            "description": e.description
        })
        response.content_type = "application/json"
        return response
    return e

# ============================================================
# ============================================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SugarCane AI — Disease & Fertilizer Advisor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --green:    #2d6a4f;
    --green-lt: #52b788;
    --cream:    #fefae0;
    --amber:    #d4a017;
    --rust:     #bc4749;
    --text:     #1a1a2e;
    --muted:    #5c5c6e;
    --card-bg:  #ffffff;
    --border:   #e0e0e0;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--cream);
    color: var(--text);
    min-height: 100vh;
  }
  header {
    background: var(--green);
    color: white;
    padding: 20px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    font-weight: 400;
  }
  header span { font-size: 2rem; }
  .subtitle { font-size: 0.85rem; color: rgba(255,255,255,0.75); margin-top: 2px; }
  .main { max-width: 960px; margin: 0 auto; padding: 32px 20px; }
  .card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }
  .card h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    font-weight: 400;
    color: var(--green);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  label { display: block; font-size: 0.8rem; font-weight: 500; color: var(--muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px; }
  input, select {
    width: 100%;
    padding: 10px 14px;
    border: 1.5px solid var(--border);
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    background: #fafafa;
    transition: border-color 0.2s;
  }
  input:focus, select:focus { outline: none; border-color: var(--green-lt); }
  .upload-area {
    border: 2px dashed var(--green-lt);
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    cursor: pointer;
    transition: background 0.2s;
    background: #f7fdf9;
  }
  .upload-area:hover { background: #edf7f0; }
  .upload-area input[type="file"] { display: none; }
  .upload-area .icon { font-size: 2.5rem; margin-bottom: 8px; }
  .upload-area p { color: var(--muted); font-size: 0.9rem; }
  #preview-img {
    max-width: 100%; border-radius: 10px; margin-top: 14px;
    display: none; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
  }
  .btn-predict {
    width: 100%; padding: 14px; background: var(--green);
    color: white; border: none; border-radius: 12px;
    font-family: 'DM Sans', sans-serif; font-size: 1rem;
    font-weight: 500; cursor: pointer; transition: background 0.2s;
    margin-top: 4px;
  }
  .btn-predict:hover  { background: #245a40; }
  .btn-predict:disabled { background: #aaa; cursor: not-allowed; }
  #result-section { display: none; }
  .disease-badge {
    display: inline-block; padding: 6px 18px;
    border-radius: 99px; font-weight: 500; font-size: 1rem;
    background: var(--green); color: white; margin-bottom: 12px;
  }
  .confidence-bar-wrap { margin: 8px 0 18px; }
  .confidence-bar-wrap span { font-size: 0.8rem; color: var(--muted); }
  .bar-track { background: #e8f5ec; border-radius: 99px; height: 10px; margin-top: 4px; }
  .bar-fill  { background: var(--green-lt); height: 100%; border-radius: 99px; transition: width 0.6s ease; }
  .prob-row { display: flex; justify-content: space-between; align-items: center; margin: 8px 0; font-size: 0.9rem; }
  .prob-label { color: var(--text); }
  .prob-val   { color: var(--green); font-weight: 500; }
  .prob-mini-bar { height: 5px; background: var(--green-lt); border-radius: 3px; margin-top: 3px; }
  .advisory-list { list-style: none; }
  .advisory-list li {
    padding: 8px 12px; background: #fff9e6; border-radius: 8px;
    margin-bottom: 8px; font-size: 0.9rem; border-left: 3px solid var(--amber);
  }
  .fert-card {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; background: #f5faf7; border-radius: 10px;
    margin-bottom: 8px; border: 1px solid #d6ede3;
  }
  .fert-name { font-weight: 500; color: var(--green); }
  .fert-prob { font-size: 0.9rem; color: var(--muted); }
  .fert-rank { font-size: 1.1rem; font-weight: 600; color: var(--amber); min-width: 26px; }
  .alpha-info {
    background: #f0f4ff; border-radius: 8px; padding: 10px 14px;
    font-size: 0.85rem; color: #3a4a7a; margin-bottom: 16px;
    border-left: 3px solid #667eea;
  }
  #gradcam-img { max-width: 100%; border-radius: 10px; margin-top: 10px; display: none; }
  .spinner {
    display: none; width: 36px; height: 36px; border: 4px solid #e0e0e0;
    border-top-color: var(--green); border-radius: 50%;
    animation: spin 0.8s linear infinite; margin: 20px auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .error-msg { background: #ffeef0; border-left: 3px solid var(--rust); border-radius: 8px; padding: 12px 16px; color: #8b0000; font-size: 0.9rem; display: none; white-space: pre-wrap; }
  @media (max-width: 600px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <span>🌿</span>
  <div>
    <h1>SugarCane AI Advisor</h1>
    <div class="subtitle">Multimodal Disease Detection & Fertilizer / Pesticide Recommendation</div>
  </div>
</header>

<div class="main">

  <div class="card">
    <h2>📷 Leaf Image</h2>
    <div class="upload-area" onclick="document.getElementById('file-input').click()">
      <input type="file" id="file-input" accept="image/*" onchange="handleFileSelect(this)">
      <div class="icon">🍃</div>
      <p>Click to upload a sugarcane leaf photo</p>
      <p style="font-size:0.75rem;margin-top:4px">JPG, PNG, WEBP supported</p>
    </div>
    <img id="preview-img" src="" alt="Preview">
  </div>

  <div class="card">
    <h2>🌡️ Field & Soil Data</h2>
    <div class="grid-3" style="margin-bottom:16px">
      <div><label>Temperature (°C)</label><input type="number" id="temp" value="28" min="0" max="50"></div>
      <div><label>Humidity (%)</label><input type="number" id="humidity" value="82" min="0" max="100"></div>
      <div><label>Moisture (%)</label><input type="number" id="moisture" value="44" min="0" max="100"></div>
    </div>
    <div class="grid-3" style="margin-bottom:16px">
      <div><label>Nitrogen (kg/ha)</label><input type="number" id="nitrogen" value="70" min="0" max="200"></div>
      <div><label>Potassium (kg/ha)</label><input type="number" id="potassium" value="45" min="0" max="200"></div>
      <div><label>Phosphorous (kg/ha)</label><input type="number" id="phosphorous" value="38" min="0" max="200"></div>
    </div>
    <div class="grid-2">
      <div>
        <label>Soil Type</label>
        <select id="soil-type">
          <option>Loamy</option><option>Clayey</option><option>Sandy</option>
          <option>Red</option><option>Black</option>
        </select>
      </div>
      <div>
        <label>Crop Type</label>
        <select id="crop-type">
          <option>Sugarcane</option><option>Maize</option><option>Wheat</option>
          <option>Rice</option><option>Cotton</option>
        </select>
      </div>
    </div>
  </div>

  <button class="btn-predict" id="predict-btn" onclick="runPrediction()">
    🔍 Analyze & Recommend
  </button>

  <div class="spinner" id="spinner"></div>
  <div class="error-msg" id="error-msg"></div>

  <div id="result-section">

    <div class="card" id="disease-card">
      <h2>🦠 Disease Prediction</h2>
      <div class="disease-badge" id="disease-name"></div>
      <div class="confidence-bar-wrap">
        <span>Confidence: <strong id="confidence-pct"></strong></span>
        <div class="bar-track"><div class="bar-fill" id="conf-bar"></div></div>
      </div>
      <div class="alpha-info" id="alpha-info"></div>
      <div id="disease-probs"></div>
    </div>

    <div class="card" id="fert-card">
      <h2>🌱 Fertilizer Recommendations</h2>
      <div id="fert-list"></div>
    </div>

    <div class="card" id="advisory-card">
      <h2>🔬 Pesticide & Management Advisory</h2>
      <ul class="advisory-list" id="advisory-list"></ul>
    </div>

    <div class="card" id="gradcam-card">
      <h2>🔍 AI Explanation (Grad-CAM)</h2>
      <p style="font-size:0.85rem;color:var(--muted);margin-bottom:8px">Heatmap shows which leaf regions influenced the prediction.</p>
      <img id="gradcam-img" src="" alt="Grad-CAM">
    </div>

  </div>
</div>

<script>
let selectedFile = null;

function handleFileSelect(input) {
  const file = input.files[0];
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('preview-img');
    img.src = e.target.result;
    img.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

async function runPrediction() {
  if (!selectedFile) { showError('Please upload a leaf image first.'); return; }

  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.textContent = 'Analyzing...';
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('result-section').style.display = 'none';

  try {
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('Temparature', document.getElementById('temp').value);
    formData.append('Humidity', document.getElementById('humidity').value);
    formData.append('Moisture', document.getElementById('moisture').value);
    formData.append('Nitrogen', document.getElementById('nitrogen').value);
    formData.append('Potassium', document.getElementById('potassium').value);
    formData.append('Phosphorous', document.getElementById('phosphorous').value);
    formData.append('Soil Type', document.getElementById('soil-type').value);
    formData.append('Crop Type', document.getElementById('crop-type').value);

const resp = await fetch('/predict', { method: 'POST', body: formData });

const rawText = await resp.text();
let data = null;

try {
  data = rawText ? JSON.parse(rawText) : null;
} catch (e) {
  showError(
    `Server returned invalid response.\n` +
    `HTTP ${resp.status}\n\n` +
    rawText.slice(0, 800)
  );
  return;
}

if (!resp.ok) {
  showError(data?.error || `Server error (HTTP ${resp.status})`);
  console.error(data);
  return;
}

if (!data) {
  showError(`Empty response from server (HTTP ${resp.status})`);
  return;
}

if (data.error) {
  showError(data.error);
  return;
}

renderResults(data);
  } catch(e) {
    showError('Network error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 Analyze & Recommend';
    document.getElementById('spinner').style.display = 'none';
  }
}

function renderResults(data) {
  document.getElementById('result-section').style.display = 'block';

  document.getElementById('disease-name').textContent = data.predicted_disease;
  const conf = (data.confidence * 100).toFixed(1);
  document.getElementById('confidence-pct').textContent = conf + '%';
  document.getElementById('conf-bar').style.width = conf + '%';
  document.getElementById('alpha-info').textContent =
    `Fusion: ${(data.fusion_alpha*100).toFixed(0)}% image weight, ${((1-data.fusion_alpha)*100).toFixed(0)}% sensor weight (confidence-adaptive)`;

  const probsDiv = document.getElementById('disease-probs');
  probsDiv.innerHTML = '';
  const sortedD = Object.entries(data.fused_disease_probs).sort((a,b)=>b[1]-a[1]);
  sortedD.forEach(([name, prob]) => {
    const pct = (prob * 100).toFixed(1);
    probsDiv.innerHTML += `
      <div class="prob-row">
        <span class="prob-label">${name}</span>
        <span class="prob-val">${pct}%</span>
      </div>
      <div class="prob-mini-bar" style="width:${pct}%;max-width:100%"></div>`;
  });

  const fertDiv = document.getElementById('fert-list');
  fertDiv.innerHTML = '';
  const fertItems = Object.entries(data.fertilizer_recommendations);
  fertItems.slice(0, 5).forEach(([name, prob], i) => {
    fertDiv.innerHTML += `
      <div class="fert-card">
        <span class="fert-rank">${i+1}</span>
        <span class="fert-name">${name}</span>
        <span class="fert-prob">${(prob*100).toFixed(1)}%</span>
      </div>`;
  });

  const advList = document.getElementById('advisory-list');
  advList.innerHTML = '';
  data.pesticide_advisory.forEach(item => {
    advList.innerHTML += `<li>${item}</li>`;
  });

  if (data.gradcam_base64) {
    const gc = document.getElementById('gradcam-img');
    gc.src = 'data:image/png;base64,' + data.gradcam_base64;
    gc.style.display = 'block';
  }

  document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = '⚠️ ' + msg;
  el.style.display = 'block';
}
</script>
</body>
</html>"""


# ============================================================
# STARTUP — load models at import time (works with --preload)
# ============================================================
import traceback as _tb

MODELS_READY = False
MODELS_ERROR = None

try:
    load_all_models()
    MODELS_READY = True
    print("[STARTUP] Models ready — app fully operational.")
except Exception as _e:
    MODELS_ERROR = _tb.format_exc()
    print(f"[STARTUP ERROR] {_e}")

# ============================================================
# FLASK ROUTES
# ============================================================

LOADING_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta http-equiv="refresh" content="10">
<title>SugarCane AI — Starting up...</title>
<style>
  body{font-family:sans-serif;background:#fefae0;display:flex;align-items:center;
       justify-content:center;min-height:100vh;margin:0;}
  .box{text-align:center;background:white;padding:40px 60px;border-radius:16px;
       box-shadow:0 4px 20px rgba(0,0,0,0.1);}
  h2{color:#2d6a4f;margin-bottom:12px;}
  p{color:#5c5c6e;font-size:0.95rem;}
  .spinner{width:40px;height:40px;border:4px solid #e0e0e0;border-top-color:#52b788;
           border-radius:50%;animation:spin 0.9s linear infinite;margin:20px auto;}
  @keyframes spin{to{transform:rotate(360deg);}}
</style></head>
<body><div class="box">
  <div class="spinner"></div>
  <h2>🌿 SugarCane AI is starting up...</h2>
  <p>Downloading AI models from Google Drive.<br>
     This takes <strong>1–3 minutes</strong> on first launch.</p>
  <p style="margin-top:12px;font-size:0.85rem;color:#aaa">
     This page refreshes automatically every 10 seconds.</p>
</div></body></html>"""

@app.route("/")
def index():
    if MODELS_ERROR:
        return f"<h2>Startup Error</h2><pre>{MODELS_ERROR}</pre>", 500
    if not MODELS_READY:
        return LOADING_PAGE, 200
    return HTML_PAGE

@app.route("/health")
def health():
    if MODELS_READY:
        return "OK", 200
    if MODELS_ERROR:
        return f"ERROR: {MODELS_ERROR}", 500
    return "Loading...", 200

@app.route("/predict", methods=["POST"])
def predict():
    if MODELS_ERROR:
        return jsonify({"error": f"Model load failed: {MODELS_ERROR}"}), 500
    if not MODELS_READY:
        return jsonify({"error": "Models are still loading. Please wait 1–2 minutes and try again."}), 503

    with PREDICT_LOCK:
        tmp_path = None
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        suffix = os.path.splitext(image_file.filename)[-1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image_file.save(tmp.name)
            tmp_path = tmp.name

        env_input = {
            "Temparature":  float(request.form.get("Temparature", 28)),
            "Humidity":     float(request.form.get("Humidity", 70)),
            "Moisture":     float(request.form.get("Moisture", 40)),
            "Nitrogen":     float(request.form.get("Nitrogen", 50)),
            "Potassium":    float(request.form.get("Potassium", 40)),
            "Phosphorous":  float(request.form.get("Phosphorous", 30)),
            "Soil Type":    request.form.get("Soil Type", "Loamy"),
            "Crop Type":    request.form.get("Crop Type", "Sugarcane"),
        }

        img_probs   = predict_image_disease_probs(tmp_path)
        env_df      = preprocess_env_input(env_input)
        env_probs   = predict_env_disease_probs(env_df)
        fused, alpha = fuse_disease_probabilities(img_probs, env_probs)
        fert_probs  = predict_final_recommendations(env_df, fused)

        predicted_disease = max(fused, key=fused.get)
        confidence        = fused[predicted_disease]
        gradcam_b64       = None

        response = {
            "predicted_disease":        str(predicted_disease),
            "confidence":               float(confidence),
            "fusion_alpha":             float(alpha),
            "image_disease_probs":      {k: float(v) for k, v in img_probs.items()},
            "environment_disease_probs":{k: float(v) for k, v in env_probs.items()},
            "fused_disease_probs":      {k: float(v) for k, v in fused.items()},
            "fertilizer_recommendations":{k: float(v) for k, v in fert_probs.items()},
            "top_fertilizer":           str(max(fert_probs, key=fert_probs.get)),
            "pesticide_advisory":       [str(x) for x in PESTICIDE_ADVISORY.get(predicted_disease, [])],
            "gradcam_base64":           gradcam_b64,
        }
        return jsonify(to_python(response))

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
