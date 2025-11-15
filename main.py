import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import uvicorn
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "exported_inference"
CLASSES_FILE = BASE_DIR / "classes.txt"
UPLOADS = BASE_DIR / "uploads"
UPLOADS.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI()

# only mount static if folder exists (avoid crashing if static/ missing)
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# load classes
if CLASSES_FILE.exists():
    CLASSES = [line.strip() for line in CLASSES_FILE.read_text(encoding="utf8").splitlines() if line.strip()]
else:
    CLASSES = None

# load savedmodel
print("Loading SavedModel from:", MODEL_DIR)
saved = tf.saved_model.load(str(MODEL_DIR))
signature = saved.signatures.get("serving_default")
try:
    sig_inputs = signature.structured_input_signature[1]
    INPUT_NAME = list(sig_inputs.keys())[0]
except Exception:
    INPUT_NAME = None


def predict_logits_from_image(image_input, target_size=(64, 64)):
    """
    Accepts a PIL Image (or numpy array) and returns raw model outputs (logits or probs).
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB").resize(target_size)
        arr = np.array(img).astype(np.float32) / 255.0
    else:
        arr = np.asarray(image_input).astype(np.float32)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)  # add batch dim
    tensor = tf.constant(arr)
    if INPUT_NAME:
        output = signature(**{INPUT_NAME: tensor})
    else:
        output = signature(tensor)
    # extract numeric array
    if isinstance(output, dict):
        key = list(output.keys())[0]
        res = output[key].numpy()
    else:
        res = np.asarray(output)
    # ensure 1D logits vector for single sample
    if res.ndim == 2 and res.shape[0] == 1:
        res = res[0]
    return res  # numpy array


def softmax_logits(logits: np.ndarray):
    # stable softmax
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    probs = exps / np.sum(exps)
    return probs


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("main.html", {"request": request, "prediction_text": ""})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    # read and save temporary file
    contents = await image.read()
    filename = image.filename
    tmp_path = UPLOADS / filename
    with open(tmp_path, "wb") as f:
        f.write(contents)

    try:
        img = Image.open(tmp_path)
        logits = predict_logits_from_image(img)  # raw outputs
        probs = softmax_logits(logits)           # convert to normalized probabilities

        # full probs list for table
        if CLASSES:
            all_probs = [(CLASSES[i], float(probs[i])) for i in range(len(probs))]
        else:
            all_probs = [(str(i), float(probs[i])) for i in range(len(probs))]

        # top-3
        topk = int(min(3, probs.shape[0]))
        top_idx = np.argsort(probs)[-topk:][::-1]
        top3 = []
        for i in top_idx:
            label = CLASSES[i] if CLASSES and i < len(CLASSES) else str(i)
            top3.append((label, float(probs[i]), int(i)))

        # highest
        best_label, best_prob, best_idx = top3[0][0], top3[0][1], top3[0][2]
        text = f'The Skin Disease is \"{best_label}\" (index={best_idx}, score={best_prob:.4f})'

    except Exception as e:
        text = f"Prediction error: {e}"
        top3 = []
        all_probs = []

    finally:
        # cleanup
        try:
            tmp_path.unlink()
        except Exception:
            pass

    return templates.TemplateResponse(
        "main.html",
        {
            "request": request,
            "prediction_text": text,
            "top3": top3,         # list of tuples (label, prob, idx)
            "all_probs": all_probs  # list of tuples (label, prob)
        },
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)