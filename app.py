import os
import threading

import torch
from flask import Flask, render_template, request, jsonify
from model import HaroonGPT, GPTConfig
from tokenizer import ByteTokenizer

app = Flask(__name__)

# تقليل استخدام CPU والذاكرة على Render
DEVICE = "cpu"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

tokenizer = ByteTokenizer()
model = None
model_status = "جاري تحميل النموذج..."
model_lock = threading.Lock()


def load_model():
    global model, model_status

    path = os.environ.get(
        "HAROON_CHECKPOINT",
        "checkpoints/model.pt"
    )

    if not os.path.isfile(path):
        model = None
        model_status = f"لم يتم العثور على ملف الأوزان: {path}"
        print(model_status)
        return

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        cfg = GPTConfig()

        for key, value in checkpoint.get("config", {}).items():
            setattr(cfg, key, value)

        model_instance = HaroonGPT(cfg)
        model_instance.load_state_dict(checkpoint["model"])
        model_instance.eval()

        model = model_instance

        step = checkpoint.get("step", "?")
        model_status = (
            f"تم تحميل Haroon-2.5-HK بنجاح (step {step})"
        )

        print(model_status)

    except Exception as exc:
        model = None
        model_status = f"فشل تحميل النموذج: {exc}"
        print(model_status)


load_model()


@app.get("/")
def home():
    return render_template(
        "index.html",
        model_status=model_status
    )


@app.get("/api/status")
def status():
    return jsonify({
        "name": "Haroon-2.5-HK",
        "ready": model is not None,
        "device": DEVICE,
        "status": model_status
    })


@app.post("/api/chat")
def chat():
    if model is None:
        return jsonify({
            "error": "النموذج غير جاهز.",
            "status": model_status
        }), 503

    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    history = body.get("history", [])

    if not message:
        return jsonify({
            "error": "اكتب رسالة أولًا."
        }), 400

    # المحافظة على سجل محادثة محدود
    prompt = ""

    if isinstance(history, list):
        for item in history[-4:]:
            if not isinstance(item, dict):
                continue

            role = (
                "المستخدم"
                if item.get("role") == "user"
                else "Haroon"
            )

            content = str(item.get("content", ""))[:300]
            prompt += f"{role}: {content}\n"

    prompt += f"المستخدم: {message[:500]}\nHaroon:"

    try:
        ids_list = tokenizer.encode(prompt)

        # الاحتفاظ بآخر جزء من السياق لتقليل التكلفة
        ids_list = ids_list[-128:]

        ids = torch.tensor(
            [ids_list],
            dtype=torch.long
        )

        # منع طلبين من تشغيل التوليد بالتزامن
        with model_lock:
            with torch.inference_mode():
                out = model.generate(
                    ids,
                    max_new_tokens=48,
                    temperature=0.8,
                    top_k=20
                )

        answer_ids = out[0, ids.shape[1]:].tolist()
        answer = tokenizer.decode(answer_ids).strip()

        for marker in ["\nالمستخدم:", "\nHaroon:"]:
            if marker in answer:
                answer = answer.split(marker)[0].strip()

        if not answer:
            answer = "لم أتمكن من توليد إجابة واضحة. حاول مرة أخرى."

        return jsonify({"answer": answer})

    except Exception as exc:
        app.logger.exception("Chat generation failed")

        return jsonify({
            "error": "حدث خطأ أثناء توليد الإجابة.",
            "details": str(exc)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
        )
