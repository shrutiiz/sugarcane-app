"""
CODE 05 - Web Application (Flask - Permanent Deployment)
Fixes applied:
  - gdown 6.x compatible download (no fuzzy argument)
  - Async job queue so Render 30s proxy timeout never triggers
  - Working file upload via proper label+addEventListener
"""

import os
import json
import base64
import tempfile
import io
import zipfile
import shutil
import traceback as _tb
import threading
import uuid
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module

from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# CONFIG
# ============================================================
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR      = os.path.join(BASE_DIR, "artifacts")
ARTIFACT_ZIP_PATH = os.path.join(BASE_DIR, "artifacts.zip")

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
    "YellowLeafDisease":   "YellowLeaf",
}

NUMERIC_COLS     = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
ENV_FEATURE_COLS = NUMERIC_COLS + ["Soil Type Enc", "Crop Type Enc"]
REC_FEATURE_COLS = ENV_FEATURE_COLS + [
    "dprob_BacterialBlight", "dprob_Healthy", "dprob_Mosaic",
    "dprob_RedRot", "dprob_Rust", "dprob_YellowLeaf"
]

PESTICIDE_ADVISORY = {
    "BacterialBlight": [
        "Apply copper-based bactericide (e.g. Blitox 50 WP) - 3g/L water",
        "Improve field drainage to reduce moisture buildup",
        "Remove and destroy severely infected cane stalks",
    ],
    "Healthy": [
        "No pesticide treatment needed",
        "Continue balanced NPK fertilization schedule",
        "Monitor for early symptom signs every 2 weeks",
    ],
    "Mosaic": [
        "Control aphid vectors using systemic insecticide (e.g. Imidacloprid)",
        "Remove and burn severely mosaic-infected plants",
        "Use certified disease-free seed cane for next crop",
    ],
    "RedRot": [
        "Apply Mancozeb or Carbendazim fungicide as soil drench",
        "Avoid waterlogging - install proper drainage channels",
        "Treat seed cane with Bavistin before planting",
    ],
    "Rust": [
        "Apply Propiconazole or Mancozeb fungicide spray",
        "Monitor fields regularly during high humidity periods",
        "Reduce plant density to improve air circulation",
    ],
    "YellowLeaf": [
        "Apply balanced fertilizer focusing on Potassium and Micronutrients",
        "Use foliar spray of 0.5% ZnSO4 + 0.5% MgSO4",
        "Monitor for whitefly and aphid vectors of YLD virus",
    ],
}

MODELS = {}

# ============================================================
# DOWNLOAD ARTIFACTS  —  gdown 6.x compatible
# ============================================================

def ensure_artifacts_downloaded():
    if os.path.exists(CNN_WEIGHTS_PATH):
        print("[INFO] artifacts/ already present, skipping download.")
        return

    print("[INFO] Downloading artifacts.zip from Google Drive...")

    if os.path.exists(ARTIFACT_DIR):
        shutil.rmtree(ARTIFACT_DIR)
    if os.path.exists(ARTIFACT_ZIP_PATH):
        os.remove(ARTIFACT_ZIP_PATH)

    import gdown
    # gdown 6.x: no fuzzy argument — use url= directly
    url = f"https://drive.google.com/uc?id={ARTIFACTS_ZIP_FILE_ID}&export=download"
    gdown.download(url, ARTIFACT_ZIP_PATH, quiet=False)

    size = os.path.getsize(ARTIFACT_ZIP_PATH)
    print(f"[INFO] Downloaded {size} bytes to {ARTIFACT_ZIP_PATH}")

    if size < 1000:
        raise RuntimeError(
            "Download too small — Google Drive may have returned an HTML error page. "
            "Make sure the file is shared as 'Anyone with the link can view'."
        )

    with zipfile.ZipFile(ARTIFACT_ZIP_PATH, "r") as zf:
        zf.extractall(BASE_DIR)

    print("[INFO] Artifacts extracted successfully.")

# ============================================================
# HELPERS
# ============================================================

def check_file_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at: {path}")

def normalize_prob_dict(prob_dict):
    keys = list(prob_dict.keys())
    vals = np.array(list(prob_dict.values()), dtype=np.float32)
    vals = np.clip(vals, 1e-9, None)
    vals /= vals.sum()
    return {k: float(v) for k, v in zip(keys, vals)}

def to_canonical_name(name):
    return CLASS_NAME_MAP.get(str(name).strip(), str(name).strip())

def apply_temperature_scaling(raw_probs, T):
    logits  = np.log(raw_probs + 1e-9)
    scaled  = logits / T
    shifted = scaled - scaled.max()
    exp_s   = np.exp(shifted)
    return exp_s / exp_s.sum()

def safe_label_transform(encoder, value, field_name):
    value = str(value).strip()
    if value not in encoder.classes_:
        raise ValueError(f"Unknown {field_name}: '{value}'. Allowed: {list(encoder.classes_)}")
    return int(encoder.transform([value])[0])

def to_python(obj):
    if isinstance(obj, dict):   return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_python(x) for x in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj

# ============================================================
# CNN MODEL
# ============================================================

def build_cnn_model(num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    attn = tf.keras.layers.Dense(1280 // 4, activation="relu",     name="attn_squeeze")(x)
    attn = tf.keras.layers.Dense(1280,       activation="sigmoid",  name="attn_excite")(attn)
    x    = tf.keras.layers.Multiply(name="attn_weighted")([x, attn])
    x    = tf.keras.layers.Dropout(0.35)(x)
    out  = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return tf.keras.Model(inputs, out, name="EfficientNet_AttentionHead")

# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_all_models():
    print("[STARTUP] Loading all models...")
    ensure_artifacts_downloaded()

    for name, path in {
        "CNN weights":        CNN_WEIGHTS_PATH,
        "Class names":        CLASS_NAMES_PATH,
        "Env disease model":  ENV_DISEASE_MODEL_PATH,
        "Env encoder":        ENV_DISEASE_ENCODER_PATH,
        "Rec model":          REC_MODEL_PATH,
        "Fert encoder":       FERT_ENCODER_PATH,
        "Soil encoder":       SOIL_ENCODER_PATH,
        "Crop encoder":       CROP_ENCODER_PATH,
        "Numeric scaler":     NUMERIC_SCALER_PATH,
    }.items():
        check_file_exists(path, name)

    with open(CLASS_NAMES_PATH) as f:
        MODELS["class_names"] = json.load(f)

    MODELS["cnn"] = build_cnn_model(len(MODELS["class_names"]))
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

    print(f"[STARTUP] Temperature T={MODELS['temperature']:.4f}")
    print("[STARTUP] All models loaded successfully.")

    # Warm up: run one dummy inference in the main thread so TF builds its graph here.
    # This prevents deadlocks when background threads later call the model.
    print("[STARTUP] Warming up CNN model...")
    dummy = tf.zeros([1, IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32)
    _ = MODELS["cnn"](dummy, training=False)
    print("[STARTUP] Warmup complete.")

# ============================================================
# INFERENCE
# ============================================================

def predict_image_disease_probs(image_path):
    img   = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr   = tf.keras.utils.img_to_array(img)
    arr   = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(arr, 0))
    arr   = tf.constant(arr, dtype=tf.float32)
    # Direct model call is thread-safe; model.predict() deadlocks in background threads
    raw   = MODELS["cnn"](arr, training=False).numpy()[0]
    cal   = apply_temperature_scaling(raw, MODELS["temperature"])
    raw_p = {cls: float(p) for cls, p in zip(MODELS["class_names"], cal)}
    canon = {d: 0.0 for d in CANONICAL_DISEASE_CLASSES}
    for rn, p in raw_p.items():
        c = to_canonical_name(rn)
        if c in canon:
            canon[c] += p
    return normalize_prob_dict(canon)

def preprocess_env_input(env_input):
    df = pd.DataFrame([env_input]).copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df[NUMERIC_COLS]     = MODELS["scaler"].transform(df[NUMERIC_COLS])
    df["Soil Type Enc"]  = safe_label_transform(MODELS["soil_encoder"], df.loc[0, "Soil Type"], "Soil Type")
    df["Crop Type Enc"]  = safe_label_transform(MODELS["crop_encoder"], df.loc[0, "Crop Type"], "Crop Type")
    return df

def predict_env_disease_probs(env_df):
    probs  = MODELS["env_model"].predict_proba(env_df[ENV_FEATURE_COLS])[0]
    names  = MODELS["env_encoder"].inverse_transform(np.arange(len(probs)))
    out    = {to_canonical_name(c): float(p) for c, p in zip(names, probs)}
    full   = {d: out.get(d, 0.0) for d in CANONICAL_DISEASE_CLASSES}
    return normalize_prob_dict(full)

def fuse_disease_probabilities(image_probs, env_probs):
    ci    = max(image_probs.values())
    ce    = max(env_probs.values())
    alpha = ci / (ci + ce + 1e-8)
    fused = {d: alpha * image_probs.get(d, 0.0) + (1 - alpha) * env_probs.get(d, 0.0)
             for d in CANONICAL_DISEASE_CLASSES}
    return normalize_prob_dict(fused), float(alpha)

def predict_final_recommendations(env_df, fused_disease_probs):
    scores = np.zeros(len(MODELS["fert_encoder"].classes_), dtype=np.float32)
    for disease, prob in fused_disease_probs.items():
        tmp = env_df.copy()
        for d in CANONICAL_DISEASE_CLASSES:
            tmp[f"dprob_{d}"] = 0.0
        tmp[f"dprob_{disease}"] = 1.0
        scores += float(prob) * MODELS["rec_model"].predict_proba(tmp[REC_FEATURE_COLS])[0]
    scores /= scores.sum()
    names = MODELS["fert_encoder"].inverse_transform(np.arange(len(scores)))
    return dict(sorted({str(k): float(v) for k, v in zip(names, scores)}.items(),
                       key=lambda x: x[1], reverse=True))

def generate_gradcam_base64(image_path):
    try:
        cnn = MODELS["cnn"]
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        img_arr = tf.keras.utils.img_to_array(img)
        proc = tf.keras.applications.efficientnet.preprocess_input(
            np.expand_dims(img_arr.copy(), 0).astype(np.float32)
        )
        tensor = tf.convert_to_tensor(proc, dtype=tf.float32)

        base_model = next(
            (l for l in cnn.layers if isinstance(l, tf.keras.Model) and "efficientnet" in l.name.lower()),
            None
        )
        if base_model is None:
            return None

        grad_model = tf.keras.Model(
            inputs=cnn.inputs,
            outputs=[base_model.get_layer("top_conv").output, cnn.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(tensor, training=False)
            class_ch = preds[:, tf.argmax(preds[0])]

        grads = tape.gradient(class_ch, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)

        heatmap_r = np.array(tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE)).squeeze()
        cmap = cm_module.get_cmap("jet")
        overlay = np.clip(0.5 * (img_arr / 255.0) + 0.5 * cmap(heatmap_r)[:, :, :3], 0, 1)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(overlay)
        ax.set_title("Grad-CAM")
        ax.axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=60, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")
        return None

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# HTML
# ============================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SugarCane AI</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root{--green:#2d6a4f;--green-lt:#52b788;--cream:#fefae0;--amber:#d4a017;--rust:#bc4749;--text:#1a1a2e;--muted:#5c5c6e;--card:#fff;--border:#e0e0e0;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'DM Sans',sans-serif;background:var(--cream);color:var(--text);min-height:100vh;}
  header{background:var(--green);color:#fff;padding:20px 40px;display:flex;align-items:center;gap:16px;}
  header h1{font-family:'DM Serif Display',serif;font-size:1.8rem;font-weight:400;}
  .subtitle{font-size:.85rem;color:rgba(255,255,255,.75);margin-top:2px;}
  .main{max-width:960px;margin:0 auto;padding:32px 20px;}
  .card{background:var(--card);border-radius:16px;padding:28px;margin-bottom:24px;border:1px solid var(--border);box-shadow:0 2px 12px rgba(0,0,0,.06);}
  .card h2{font-family:'DM Serif Display',serif;font-size:1.3rem;font-weight:400;color:var(--green);margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid var(--border);}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
  .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}
  .field-label{display:block;font-size:.8rem;font-weight:500;color:var(--muted);margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px;}
  input[type=number],select{width:100%;padding:10px 14px;border:1.5px solid var(--border);border-radius:10px;font-family:'DM Sans',sans-serif;font-size:.95rem;background:#fafafa;transition:border-color .2s;}
  input[type=number]:focus,select:focus{outline:none;border-color:var(--green-lt);}
  /* FILE UPLOAD */
  #file-input{width:1px;height:1px;opacity:0;position:absolute;}
  .upload-label{display:block;border:2px dashed var(--green-lt);border-radius:12px;padding:32px;text-align:center;cursor:pointer;transition:background .2s;background:#f7fdf9;user-select:none;}
  .upload-label:hover{background:#edf7f0;}
  .upload-icon{font-size:2.5rem;margin-bottom:8px;}
  .upload-label p{color:var(--muted);font-size:.9rem;}
  #file-status{margin-top:8px;font-size:.85rem;font-weight:500;color:var(--green);display:none;text-align:center;}
  #preview-img{max-width:100%;border-radius:10px;margin-top:14px;display:none;box-shadow:0 2px 8px rgba(0,0,0,.12);}
  /* BUTTON */
  .btn{width:100%;padding:14px;background:var(--green);color:#fff;border:none;border-radius:12px;font-family:'DM Sans',sans-serif;font-size:1rem;font-weight:500;cursor:pointer;transition:background .2s;margin-top:4px;}
  .btn:hover{background:#245a40;}
  .btn:disabled{background:#aaa;cursor:not-allowed;}
  /* SPINNER */
  .spinner{display:none;width:36px;height:36px;border:4px solid #e0e0e0;border-top-color:var(--green);border-radius:50%;animation:spin .8s linear infinite;margin:20px auto;}
  @keyframes spin{to{transform:rotate(360deg);}}
  .status-msg{text-align:center;color:var(--muted);font-size:.9rem;display:none;padding:8px;}
  .error-msg{background:#ffeef0;border-left:3px solid var(--rust);border-radius:8px;padding:12px 16px;color:#8b0000;font-size:.9rem;display:none;white-space:pre-wrap;margin-top:12px;}
  /* RESULTS */
  #result-section{display:none;}
  .disease-badge{display:inline-block;padding:6px 18px;border-radius:99px;font-weight:500;background:var(--green);color:#fff;margin-bottom:12px;}
  .bar-wrap{margin:8px 0 18px;}
  .bar-wrap span{font-size:.8rem;color:var(--muted);}
  .bar-track{background:#e8f5ec;border-radius:99px;height:10px;margin-top:4px;}
  .bar-fill{background:var(--green-lt);height:100%;border-radius:99px;transition:width .6s;}
  .prob-row{display:flex;justify-content:space-between;margin:8px 0;font-size:.9rem;}
  .prob-mini{height:5px;background:var(--green-lt);border-radius:3px;margin-top:3px;}
  .fert-row{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:#f5faf7;border-radius:10px;margin-bottom:8px;border:1px solid #d6ede3;}
  .fert-rank{font-size:1.1rem;font-weight:600;color:var(--amber);min-width:26px;}
  .fert-name{font-weight:500;color:var(--green);}
  .fert-pct{font-size:.9rem;color:var(--muted);}
  .alpha-box{background:#f0f4ff;border-radius:8px;padding:10px 14px;font-size:.85rem;color:#3a4a7a;margin-bottom:16px;border-left:3px solid #667eea;}
  .adv-list{list-style:none;}
  .adv-list li{padding:8px 12px;background:#fff9e6;border-radius:8px;margin-bottom:8px;font-size:.9rem;border-left:3px solid var(--amber);}
  #gradcam-card{display:none;}
  #gradcam-img{max-width:100%;border-radius:10px;margin-top:10px;}
  @media(max-width:600px){.grid-2,.grid-3{grid-template-columns:1fr;}header{padding:16px 20px;}}
</style>
</head>
<body>
<header>
  <span>&#127807;</span>
  <div>
    <h1>SugarCane AI Advisor</h1>
    <div class="subtitle">Multimodal Disease Detection &amp; Fertilizer / Pesticide Recommendation</div>
  </div>
</header>

<div class="main">

  <div class="card">
    <h2>&#128247; Leaf Image</h2>
    <input type="file" id="file-input" accept="image/jpeg,image/jpg,image/png,image/webp">
    <label for="file-input" class="upload-label">
      <div class="upload-icon">&#127807;</div>
      <p><strong>Click here</strong> to choose a leaf photo</p>
      <p style="font-size:.75rem;margin-top:4px;color:#999">JPG / PNG / WEBP</p>
    </label>
    <div id="file-status"></div>
    <img id="preview-img" alt="preview">
  </div>

  <div class="card">
    <h2>&#127777; Field &amp; Soil Data</h2>
    <div class="grid-3" style="margin-bottom:16px">
      <div><span class="field-label">Temperature (C)</span><input type="number" id="temp"  value="28" min="0" max="50"></div>
      <div><span class="field-label">Humidity (%)</span>   <input type="number" id="hum"   value="82" min="0" max="100"></div>
      <div><span class="field-label">Moisture (%)</span>   <input type="number" id="moist" value="44" min="0" max="100"></div>
    </div>
    <div class="grid-3" style="margin-bottom:16px">
      <div><span class="field-label">Nitrogen (kg/ha)</span>  <input type="number" id="n" value="70" min="0" max="200"></div>
      <div><span class="field-label">Potassium (kg/ha)</span> <input type="number" id="k" value="45" min="0" max="200"></div>
      <div><span class="field-label">Phosphorous (kg/ha)</span><input type="number" id="p" value="38" min="0" max="200"></div>
    </div>
    <div class="grid-2">
      <div>
        <span class="field-label">Soil Type</span>
        <select id="soil"><option>Loamy</option><option>Clayey</option><option>Sandy</option><option>Red</option><option>Black</option></select>
      </div>
      <div>
        <span class="field-label">Crop Type</span>
        <select id="crop"><option>Sugarcane</option><option>Maize</option><option>Wheat</option><option>Rice</option><option>Cotton</option></select>
      </div>
    </div>
  </div>

  <button class="btn" id="btn" onclick="runPrediction()">&#128269; Analyze &amp; Recommend</button>

  <div class="spinner" id="spinner"></div>
  <div class="status-msg" id="status"></div>
  <div class="error-msg"  id="errmsg"></div>

  <div id="result-section">
    <div class="card">
      <h2>&#129440; Disease Prediction</h2>
      <div class="disease-badge" id="dis-name"></div>
      <div class="bar-wrap">
        <span>Confidence: <strong id="conf-txt"></strong></span>
        <div class="bar-track"><div class="bar-fill" id="conf-bar"></div></div>
      </div>
      <div class="alpha-box" id="alpha-box"></div>
      <div id="dis-probs"></div>
    </div>
    <div class="card">
      <h2>&#127807; Fertilizer Recommendations</h2>
      <div id="fert-list"></div>
    </div>
    <div class="card">
      <h2>&#128300; Pesticide &amp; Management Advisory</h2>
      <ul class="adv-list" id="adv-list"></ul>
    </div>
    <div class="card" id="gradcam-card">
      <h2>&#128269; AI Explanation (Grad-CAM)</h2>
      <img id="gradcam-img" alt="Grad-CAM">
    </div>
  </div>
</div>

<script>
(function(){
  var selectedFile = null;
  var pollTimer    = null;

  // ---- file input ----
  var fileInput = document.getElementById('file-input');
  fileInput.addEventListener('change', function(){
    if (!this.files || !this.files.length) return;
    selectedFile = this.files[0];
    var st = document.getElementById('file-status');
    st.textContent = 'Selected: ' + selectedFile.name;
    st.style.display = 'block';
    var rdr = new FileReader();
    rdr.onload = function(e){
      var img = document.getElementById('preview-img');
      img.src = e.target.result;
      img.style.display = 'block';
    };
    rdr.readAsDataURL(selectedFile);
  });

  function setStatus(msg){
    var el = document.getElementById('status');
    el.textContent = msg;
    el.style.display = msg ? 'block' : 'none';
  }
  function showError(msg){
    var el = document.getElementById('errmsg');
    el.textContent = '\u26A0 ' + msg;
    el.style.display = 'block';
    setStatus('');
  }
  function resetUI(){
    document.getElementById('spinner').style.display = 'none';
    var btn = document.getElementById('btn');
    btn.disabled = false;
    btn.textContent = '\uD83D\uDD0D Analyze & Recommend';
    setStatus('');
    if (pollTimer){ clearInterval(pollTimer); pollTimer = null; }
  }

  window.runPrediction = function(){
    if (!selectedFile){
      showError('Please select a leaf image first.');
      return;
    }
    var btn = document.getElementById('btn');
    btn.disabled = true;
    btn.textContent = 'Submitting...';
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('errmsg').style.display  = 'none';
    document.getElementById('result-section').style.display = 'none';
    setStatus('Uploading image...');

    var fd = new FormData();
    fd.append('image',       selectedFile);
    fd.append('Temparature', document.getElementById('temp').value);
    fd.append('Humidity',    document.getElementById('hum').value);
    fd.append('Moisture',    document.getElementById('moist').value);
    fd.append('Nitrogen',    document.getElementById('n').value);
    fd.append('Potassium',   document.getElementById('k').value);
    fd.append('Phosphorous', document.getElementById('p').value);
    fd.append('Soil Type',   document.getElementById('soil').value);
    fd.append('Crop Type',   document.getElementById('crop').value);

fetch('/submit', {method:'POST', body:fd})
  .then(function(r){
    return r.text().then(function(text){
      return { ok: r.ok, status: r.status, text: text };
    });
  })
  .then(function(resp){
    var d = null;

    try {
      d = resp.text ? JSON.parse(resp.text) : null;
    } catch (e) {
      showError('Server returned invalid response (HTTP ' + resp.status + ').');
      resetUI();
      return;
    }

    if (!resp.ok) {
      showError((d && d.error) ? d.error : ('Server error (HTTP ' + resp.status + ')'));
      resetUI();
      return;
    }

    if (!d) {
      showError('Empty response from server.');
      resetUI();
      return;
    }

    if (d.error) {
      showError(d.error);
      resetUI();
      return;
    }

    var jobId = d.job_id;
    var elapsed = 0;
    btn.textContent = 'Analyzing...';
    setStatus('AI is analyzing your image...');

    pollTimer = setInterval(function(){
      elapsed += 3;
      setStatus('Analyzing... (' + elapsed + 's elapsed)');
    
      fetch('/result/' + jobId)
        .then(function(r){
          return r.text().then(function(text){
            return { ok: r.ok, status: r.status, text: text };
          });
        })
        .then(function(resp){
          var poll = null;
    
          try {
            poll = resp.text ? JSON.parse(resp.text) : null;
          } catch (e) {
            clearInterval(pollTimer); pollTimer = null;
            resetUI();
            showError('Server returned invalid response (HTTP ' + resp.status + ').');
            return;
          }
    
          if (!resp.ok) {
            clearInterval(pollTimer); pollTimer = null;
            resetUI();
            showError((poll && poll.error) ? poll.error : ('Server error (HTTP ' + resp.status + ')'));
            return;
          }
    
          if (!poll) {
            clearInterval(pollTimer); pollTimer = null;
            resetUI();
            showError('Empty response from server.');
            return;
          }
    
          if (poll.status === 'done') {
            clearInterval(pollTimer); pollTimer = null;
            resetUI();
            renderResults(poll.result);
          } else if (poll.status === 'error') {
            clearInterval(pollTimer); pollTimer = null;
            resetUI();
            showError(poll.error || 'Analysis failed.');
          }
        })
        .catch(function(){
          clearInterval(pollTimer); pollTimer = null;
          resetUI();
          showError('Network error while polling result.');
        });
    
    }, 3000);
      })
      .catch(function(e){ showError('Network error: ' + e.message); resetUI(); });
  };

  function renderResults(data){
    document.getElementById('result-section').style.display = 'block';
    document.getElementById('dis-name').textContent = data.predicted_disease;
    var conf = (data.confidence * 100).toFixed(1);
    document.getElementById('conf-txt').textContent = conf + '%';
    document.getElementById('conf-bar').style.width  = conf + '%';
    document.getElementById('alpha-box').textContent =
      'Fusion: ' + (data.fusion_alpha*100).toFixed(0) + '% image / ' +
      ((1-data.fusion_alpha)*100).toFixed(0) + '% sensor (confidence-adaptive)';

    var pd2 = document.getElementById('dis-probs');
    pd2.innerHTML = '';
    Object.entries(data.fused_disease_probs)
      .sort(function(a,b){return b[1]-a[1];})
      .forEach(function(item){
        var pct = (item[1]*100).toFixed(1);
        pd2.innerHTML +=
          '<div class="prob-row"><span>'+item[0]+'</span><span style="color:var(--green);font-weight:500">'+pct+'%</span></div>'+
          '<div class="prob-mini" style="width:'+pct+'%;max-width:100%"></div>';
      });

    var fl = document.getElementById('fert-list');
    fl.innerHTML = '';
    Object.entries(data.fertilizer_recommendations).slice(0,5)
      .forEach(function(item,i){
        fl.innerHTML +=
          '<div class="fert-row"><span class="fert-rank">'+(i+1)+'</span>'+
          '<span class="fert-name">'+item[0]+'</span>'+
          '<span class="fert-pct">'+(item[1]*100).toFixed(1)+'%</span></div>';
      });

    var al = document.getElementById('adv-list');
    al.innerHTML = '';
    data.pesticide_advisory.forEach(function(x){ al.innerHTML += '<li>'+x+'</li>'; });

if (data.gradcam_base64){
  document.getElementById('gradcam-card').style.display = 'block';
  document.getElementById('gradcam-img').src = 'data:image/png;base64,' + data.gradcam_base64;
}

// Load Grad-CAM separately after main result is shown
if (data.job_id) {
  fetch('/gradcam/' + data.job_id)
    .then(function(r){
      return r.text().then(function(text){
        return { ok: r.ok, status: r.status, text: text };
      });
    })
    .then(function(resp){
      var gc = null;
      try {
        gc = resp.text ? JSON.parse(resp.text) : null;
      } catch (e) {
        return;
      }

      if (!resp.ok || !gc) return;

      if (gc.gradcam_base64) {
        document.getElementById('gradcam-card').style.display = 'block';
        document.getElementById('gradcam-img').src = 'data:image/png;base64,' + gc.gradcam_base64;
      }
    })
    .catch(function(){});
}
    document.getElementById('result-section').scrollIntoView({behavior:'smooth'});
  }
})();
</script>
</body>
</html>"""

# ============================================================
# JOB QUEUE  —  async so Render 30s proxy never triggers
# ============================================================
_jobs      = {}
_jobs_lock = threading.Lock()
JOB_TIMEOUT_SECONDS = 180  # mark job failed if it takes longer than this

def _watchdog(job_id, timeout):
    """Background timer — marks job as error if it never completes."""
    import time
    time.sleep(timeout)
    with _jobs_lock:
        if _jobs.get(job_id, {}).get("status") == "pending":
            _jobs[job_id] = {
                "status": "error",
                "error": f"Analysis timed out after {timeout}s. "
                         "The server CPU may be overloaded. Please try again."
            }

def _run_job(job_id, image_bytes, image_suffix, env_input):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        img_probs    = predict_image_disease_probs(tmp_path)
        env_df       = preprocess_env_input(env_input)
        env_probs    = predict_env_disease_probs(env_df)
        fused, alpha = fuse_disease_probabilities(img_probs, env_probs)
        fert_probs   = predict_final_recommendations(env_df, fused)
        predicted    = max(fused, key=fused.get)
        gradcam = None
        saved_input_path = tmp_path

        result = {
            "job_id":                      job_id,
            "predicted_disease":          str(predicted),
            "confidence":                 float(fused[predicted]),
            "fusion_alpha":               float(alpha),
            "image_disease_probs":        {k: float(v) for k, v in img_probs.items()},
            "environment_disease_probs":  {k: float(v) for k, v in env_probs.items()},
            "fused_disease_probs":        {k: float(v) for k, v in fused.items()},
            "fertilizer_recommendations": {k: float(v) for k, v in fert_probs.items()},
            "top_fertilizer":             str(max(fert_probs, key=fert_probs.get)),
            "pesticide_advisory":         [str(x) for x in PESTICIDE_ADVISORY.get(predicted, [])],
            "gradcam_base64":             None,
            "input_image_path":           saved_input_path,
        }
        with _jobs_lock:
            # Only write result if watchdog hasn't already timed it out
            if _jobs.get(job_id, {}).get("status") == "pending":
                _jobs[job_id] = {"status": "done", "result": to_python(result)}

    except Exception as e:
        with _jobs_lock:
            if _jobs.get(job_id, {}).get("status") == "pending":
                _jobs[job_id] = {"status": "error", "error": str(e), "traceback": _tb.format_exc()}
    finally:
        pass

# ============================================================
# STARTUP  —  blocking load before gunicorn forks workers
# ============================================================
MODELS_READY = False
MODELS_ERROR = None

try:
    load_all_models()
    MODELS_READY = True
    print("[STARTUP] Models ready — app fully operational.")
except Exception as _e:
    MODELS_ERROR = _tb.format_exc()
    print("[STARTUP ERROR]", _e)

# ============================================================
# ROUTES
# ============================================================

LOADING_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta http-equiv="refresh" content="8">
<title>SugarCane AI - Starting up</title>
<style>
  body{font-family:sans-serif;background:#fefae0;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;}
  .box{text-align:center;background:#fff;padding:40px 60px;border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,.1);}
  h2{color:#2d6a4f;margin-bottom:12px;} p{color:#5c5c6e;font-size:.95rem;}
  .sp{width:40px;height:40px;border:4px solid #e0e0e0;border-top-color:#52b788;border-radius:50%;animation:spin .9s linear infinite;margin:20px auto;}
  @keyframes spin{to{transform:rotate(360deg);}}
</style></head>
<body><div class="box">
  <div class="sp"></div>
  <h2>SugarCane AI is starting up...</h2>
  <p>Downloading AI models. Takes <strong>1-3 minutes</strong> on first launch.</p>
  <p style="margin-top:12px;font-size:.85rem;color:#aaa">Page refreshes every 8 seconds.</p>
</div></body></html>"""


@app.route("/")
def index():
    if MODELS_ERROR:
        return "<h2>Startup Error</h2><pre>" + MODELS_ERROR + "</pre>", 500
    if not MODELS_READY:
        return LOADING_PAGE, 200
    return HTML_PAGE


@app.route("/health")
def health():
    if MODELS_READY:  return "OK", 200
    if MODELS_ERROR:  return "ERROR: " + MODELS_ERROR, 500
    return "Loading...", 200




@app.route("/submit", methods=["POST"])
def submit():
    if MODELS_ERROR:
        return jsonify({"error": "Model load failed: " + MODELS_ERROR}), 500
    if not MODELS_READY:
        return jsonify({"error": "Models still loading — please wait and retry."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file received"}), 400
    f = request.files["image"]
    if not f or f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_bytes  = f.read()
    image_suffix = os.path.splitext(f.filename)[-1].lower() or ".jpg"

    env_input = {
        "Temparature": float(request.form.get("Temparature", 28)),
        "Humidity":    float(request.form.get("Humidity",    70)),
        "Moisture":    float(request.form.get("Moisture",    40)),
        "Nitrogen":    float(request.form.get("Nitrogen",    50)),
        "Potassium":   float(request.form.get("Potassium",   40)),
        "Phosphorous": float(request.form.get("Phosphorous", 30)),
        "Soil Type":   request.form.get("Soil Type",  "Loamy"),
        "Crop Type":   request.form.get("Crop Type",  "Sugarcane"),
    }

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending"}

    threading.Thread(target=_run_job,   args=(job_id, image_bytes, image_suffix, env_input), daemon=True).start()
    threading.Thread(target=_watchdog,  args=(job_id, JOB_TIMEOUT_SECONDS),                  daemon=True).start()

    return jsonify({"job_id": job_id})


@app.route("/result/<job_id>")
def result(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"status": "error", "error": "Unknown job ID"}), 404
    return jsonify(job)

@app.route("/gradcam/<job_id>")
def gradcam(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Unknown job ID"}), 404

    if job.get("status") != "done":
        return jsonify({"error": "Result not ready yet"}), 400

    result = job.get("result", {})
    image_path = result.get("input_image_path")
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Image not available for Grad-CAM"}), 404

    try:
        gradcam_b64 = generate_gradcam_base64(image_path)
        if os.path.exists(image_path):
            os.unlink(image_path)
        return jsonify({"gradcam_base64": gradcam_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("[INFO] Starting on http://0.0.0.0:" + str(port))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
