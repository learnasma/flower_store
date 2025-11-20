from flask import Flask, request, jsonify
import requests
import base64
import torch

app = Flask(__name__)

# رابط Jina API
JINA_URL = "https://api.clip.jina.ai"

# تحميل الـ embeddings المخزنة مسبقًا على السيرفر
# هذه الملفات تحتوي على 'names' و 'features' لكل صورة في المتجر
data = torch.load("image_features.pt")
image_names = data["names"]
image_features = data["features"]  # tensor
device = "cuda" if torch.cuda.is_available() else "cpu"
image_features = image_features.to(device)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # استلام الصورة من Flutter
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        image = request.files["image"].read()

        # تحويل الصورة ل Base64
        img_base64 = base64.b64encode(image).decode("utf-8")

        # إرسال الصورة إلى Jina API للحصول على embedding
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer naa"  # اتركيها بدون مفتاح
        }
        payload = {
            "model": "clip-vit-base-patch32",
            "input": img_base64
        }
        response = requests.post(f"{JINA_URL}/encode", json=payload, headers=headers)
        if response.status_code != 200:
            return jsonify({"error": "API error", "details": response.text}), 500

        result = response.json()
        query_feature = torch.tensor(result["data"][0]["embedding"]).to(device)
        query_feature /= query_feature.norm()  # normalize

        # حساب التشابه بين الصورة المدخلة والـ embeddings المخزنة
        similarities = (image_features @ query_feature.T).squeeze(0)
        top_k = min(3, similarities.shape[0])
        top_vals, top_idxs = similarities.topk(top_k)

        # جلب أسماء أقرب 3 صور
        top_names = [image_names[int(idx)] for idx in top_idxs]

        return jsonify({"results": top_names})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Server is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
