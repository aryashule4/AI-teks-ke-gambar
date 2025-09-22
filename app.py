import os
from io import BytesIO
from pathlib import Path
from PIL import Image
import base64

from flask import Flask, render_template, request, jsonify
import torch

app = Flask(__name__)

# Hugging Face tokens / config
HF_TOKEN = os.getenv("HF_TOKEN", None)
MODEL_T2I = os.getenv("MODEL_T2I", "runwayml/stable-diffusion-v1-5")
MODEL_I2T = os.getenv("MODEL_I2T", "Salesforce/blip-image-captioning-base")

# Load image-to-text (captioning) model (transformers pipeline)
try:
    from transformers import pipeline
    captioner = pipeline("image-to-text", model=MODEL_I2T, device=0 if torch.cuda.is_available() else -1)
    print("Captioner ready", MODEL_I2T)
except Exception as e:
    captioner = None
    print("Warning: Captioner failed to load:", e)

# Load text-to-image (diffusers) pipeline lazily to avoid long startup on Render free
pipe = None

def load_t2i():
    global pipe
    if pipe is not None:
        return pipe
    try:
        from diffusers import StableDiffusionPipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_T2I,
            use_auth_token=HF_TOKEN,
            torch_dtype=dtype,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        # disable safety checker (for demo only) — in prod, implement moderation
        try:
            pipe.safety_checker = None
        except Exception:
            pass
        print("T2I pipeline ready", MODEL_T2I)
    except Exception as e:
        print("Warning: failed to load T2I pipeline:", e)
        pipe = None
    return pipe


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/caption", methods=["POST"])
def caption_api():
    if captioner is None:
        return jsonify({"error": "Captioning model not loaded on server."}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    file = request.files['image']
    pil = Image.open(file.stream).convert('RGB')
    # run captioner
    try:
        results = captioner(pil)
        # pipeline returns list of dicts or strings depending on version
        if isinstance(results, list) and len(results) > 0:
            text = results[0].get('generated_text') if isinstance(results[0], dict) else results[0]
        else:
            text = str(results)
        return jsonify({"caption": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate_api():
    prompt = request.form.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt required."}), 400

    # load pipeline lazily
    pipeline = load_t2i()
    if pipeline is None:
        return jsonify({"error": "Text-to-image pipeline not available on server."}), 500

    guidance = float(request.form.get('guidance', 7.5))
    steps = int(request.form.get('steps', 25))

    try:
        result = pipeline(prompt, guidance_scale=guidance, num_inference_steps=steps)
        image = result.images[0]
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return jsonify({"image": b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
