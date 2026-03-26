"""
Handwritten Digit Recognizer  –  Flask web app
- Open http://127.0.0.1:5000 in your browser
- Draw a digit (0-9) on the canvas with your mouse or trackpad
- The MLP model predicts which digit you drew and shows confidence bars
- Uses scikit-learn MLP classifier trained on the sklearn digits dataset
"""

import io
import base64
import threading
import warnings
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 모델 트레이닝
# ─────────────────────────────────────────────

_model = None
_scaler = None
_accuracy = 0.0
_model_ready = False


def train_model():
    """Train MLP on sklearn digits dataset (8x8 images, pixel range 0–16)."""
    global _model, _scaler, _accuracy, _model_ready
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train, y_train)

    _model = model
    _scaler = scaler
    _accuracy = model.score(X_test, y_test)
    _model_ready = True
    print(f"[Model] Ready  |  Test accuracy: {_accuracy*100:.1f}%")


# Train in background so Flask starts immediately
threading.Thread(target=train_model, daemon=True).start()

# ─────────────────────────────────────────────
# 플라스크 앱
# ─────────────────────────────────────────────

app = Flask(__name__)

# ── Inline HTML/CSS/JS 템플릿 ──────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Digit Recognizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1e1e2e; color: #cdd6f4;
    font-family: Helvetica, Arial, sans-serif;
    display: flex; flex-direction: column;
    align-items: center; min-height: 100vh;
    padding: 32px 16px;
  }
  h1 { font-size: 1.8rem; margin-bottom: 6px; }
  p.sub { color: #a6adc8; font-size: 0.95rem; margin-bottom: 20px; }
  #status { color: #fab387; font-size: 0.88rem; margin-bottom: 16px; min-height: 18px; }

  #canvas-wrap {
    border: 3px solid #313244; border-radius: 6px; line-height: 0;
  }
  canvas {
    display: block; background: #000;
    cursor: crosshair; border-radius: 3px;
    touch-action: none;
  }

  .btn-row { display: flex; gap: 12px; margin-top: 16px; }
  button {
    padding: 8px 28px; border: none; border-radius: 6px;
    font-size: 1rem; cursor: pointer; font-weight: bold;
  }
  #btn-predict { background: #89b4fa; color: #1e1e2e; }
  #btn-predict:hover { background: #74c7ec; }
  #btn-clear   { background: #45475a; color: #cdd6f4; }
  #btn-clear:hover { background: #585b70; }

  #result {
    font-size: 2rem; font-weight: bold; color: #a6e3a1;
    margin: 20px 0 6px;
    min-height: 2.4rem;
  }

  #bars { width: 340px; margin-top: 10px; }
  .bar-row {
    display: flex; align-items: center;
    gap: 8px; margin-bottom: 5px;
  }
  .bar-digit { width: 14px; font-weight: bold; font-size: 0.9rem; }
  .bar-bg {
    flex: 1; height: 16px; background: #313244; border-radius: 3px; overflow: hidden;
  }
  .bar-fill {
    height: 100%; width: 0%; border-radius: 3px;
    transition: width 0.2s ease;
  }
  .bar-pct { width: 42px; text-align: right; font-size: 0.82rem; color: #a6adc8; }
</style>
</head>
<body>

<h1>✏ Digit Recognizer</h1>
<p class="sub">Draw a digit (0–9) in the box below</p>
<div id="status">⏳ Training model, please wait…</div>

<div id="canvas-wrap">
  <canvas id="cv" width="280" height="280"></canvas>
</div>

<div class="btn-row">
  <button id="btn-predict">Predict</button>
  <button id="btn-clear">Clear</button>
</div>

<div id="result">Draw a digit to start!</div>

<div id="bars">
  <div style="font-weight:bold;font-size:0.9rem;margin-bottom:8px;color:#cdd6f4;">
    Prediction Confidence
  </div>
</div>

<script>
// ── Canvas setup ────────────────────────────
const cv  = document.getElementById("cv");
const ctx = cv.getContext("2d");
ctx.fillStyle = "#000";
ctx.fillRect(0, 0, cv.width, cv.height);

let drawing = false;
const BRUSH = 12;

function getPos(e) {
  const r = cv.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return { x: src.clientX - r.left, y: src.clientY - r.top };
}

function paint(e) {
  if (!drawing) return;
  e.preventDefault();
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.arc(x, y, BRUSH, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
}

cv.addEventListener("mousedown",  e => { drawing = true; paint(e); });
cv.addEventListener("mousemove",  paint);
cv.addEventListener("mouseup",    () => { drawing = false; predict(); });
cv.addEventListener("mouseleave", () => { drawing = false; });
cv.addEventListener("touchstart", e => { drawing = true; paint(e); }, { passive: false });
cv.addEventListener("touchmove",  paint, { passive: false });
cv.addEventListener("touchend",   () => { drawing = false; predict(); });

// ── Bar chart init ──────────────────────────
const barsDiv = document.getElementById("bars");
const fillEls = [];
const pctEls  = [];

for (let d = 0; d < 10; d++) {
  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML = `
    <span class="bar-digit">${d}</span>
    <div class="bar-bg"><div class="bar-fill" id="fill${d}"></div></div>
    <span class="bar-pct" id="pct${d}">0%</span>`;
  barsDiv.appendChild(row);
  fillEls.push(document.getElementById("fill"+d));
  pctEls.push(document.getElementById("pct"+d));
}

// ── Predict ─────────────────────────────────
async function predict() {
  const imgData = cv.toDataURL("image/png");

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imgData })
  });
  const data = await res.json();

  if (data.error) {
    document.getElementById("result").textContent = data.error;
    return;
  }

  document.getElementById("result").textContent =
    `Prediction: ${data.predicted}  (${(data.probs[data.predicted]*100).toFixed(1)}%)`;

  for (let d = 0; d < 10; d++) {
    const p = data.probs[d];
    fillEls[d].style.width  = (p * 100).toFixed(1) + "%";
    fillEls[d].style.background = d === data.predicted ? "#a6e3a1" : "#89b4fa";
    pctEls[d].textContent   = (p * 100).toFixed(1) + "%";
  }
}

// ── Clear ────────────────────────────────────
document.getElementById("btn-clear").addEventListener("click", () => {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, cv.width, cv.height);
  document.getElementById("result").textContent = "Draw a digit to start!";
  for (let d = 0; d < 10; d++) {
    fillEls[d].style.width = "0%";
    pctEls[d].textContent  = "0%";
  }
});

document.getElementById("btn-predict").addEventListener("click", predict);

// ── Poll for model-ready status ──────────────
async function pollStatus() {
  const res  = await fetch("/status");
  const data = await res.json();
  const el   = document.getElementById("status");
  if (data.ready) {
    el.textContent = `✅ Model ready  |  Test accuracy: ${data.accuracy}%`;
    el.style.color = "#a6e3a1";
  } else {
    el.textContent = "⏳ Training model, please wait…";
    setTimeout(pollStatus, 800);
  }
}
pollStatus();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# 루트
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/status")
def status():
    return jsonify(
        ready=_model_ready,
        accuracy=f"{_accuracy*100:.1f}" if _model_ready else None
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a base64-encoded PNG from the browser canvas,
    resizes it to 8x8, rescales pixels to [0,16], and runs inference.
    """
    if not _model_ready:
        return jsonify(error="Model is still training, please wait…")

    data = request.json.get("image", "")
    # Strip "data:image/png;base64," prefix
    if "," in data:
        data = data.split(",", 1)[1]

    # Decode and open as grayscale PIL image
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # Resize to 8x8 (sklearn digits format)
    img = img.resize((8, 8), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64)

    # Rescale from [0,255] to [0,16] to match training data
    arr = arr / 255.0 * 16.0
    flat = arr.flatten().reshape(1, -1)           # shape (1, 64)
    flat_scaled = _scaler.transform(flat)

    probs = _model.predict_proba(flat_scaled)[0]  # shape (10,)
    predicted = int(np.argmax(probs))

    return jsonify(predicted=predicted, probs=probs.tolist())


# ─────────────────────────────────────────────
# 엔트리 포인트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Open http://127.0.0.1:8080 in your browser")
    app.run(debug=False, port=8080)
