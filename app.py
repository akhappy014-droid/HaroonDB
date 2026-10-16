import os
import torch
from flask import Flask, render_template, request, jsonify
from model import HaroonGPT, GPTConfig
from tokenizer import ByteTokenizer

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = ByteTokenizer()
model = None
model_status = "No trained checkpoint yet. Train locally, then deploy checkpoints/model.pt."

def load_model():
    global model, model_status
    path = os.environ.get("HAROON_CHECKPOINT", "checkpoints/model.pt")
    if not os.path.exists(path):
        model = None
        model_status = "Model weights are not installed yet. This app is ready, but train the model first."
        return
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = GPTConfig()
    for key, value in checkpoint.get("config", {}).items():
        setattr(cfg, key, value)
    model = HaroonGPT(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model_status = f"Trained Haroon-2.5-HK loaded (step {checkpoint.get('step', '?')})."

load_model()

@app.get("/")
def home():
    return render_template("index.html", model_status=model_status)

@app.get("/api/status")
def status():
    return jsonify({"name": "Haroon-2.5-HK", "ready": model is not None,
                    "device": DEVICE, "status": model_status})

@app.post("/api/chat")
def chat():
    if model is None:
        return jsonify({
            "error": "لم يتم تدريب النموذج أو تثبيت أوزانه بعد. شغّل train.py أولًا ثم انشر checkpoints/model.pt.",
            "status": model_status
        }), 503
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    history = body.get("history", [])
    if not message:
        return jsonify({"error": "اكتب رسالة أولًا."}), 400
    # Simple text-only prompt. Model quality depends on training data.
    prompt = ""
    for item in history[-6:]:
        role = "المستخدم" if item.get("role") == "user" else "Haroon"
        prompt += f"{role}: {str(item.get('content',''))[:1000]}\n"
    prompt += f"المستخدم: {message}\nHaroon:"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=220, temperature=0.8, top_k=40)
    answer = tokenizer.decode(out[0, ids.shape[1]:].tolist()).strip()
    # Stop at a later role marker if generated.
    for marker in ["\nالمستخدم:", "\nHaroon:"]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()
    return jsonify({"answer": answer or "لم أتمكن من توليد إجابة واضحة."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
