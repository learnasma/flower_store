from flask import Flask, request, jsonify
import requests
import base64

app = Flask(__name__)

JINA_URL = "https://api.clip.jina.ai"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # استلام الصورة من Flutter
        image = request.files["image"].read()

        # تحويل الصورة ل Base64
        img_base64 = base64.b64encode(image).decode("utf-8")

        # إرسالها إلى Jina API
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer naa"  # مفتوح بدون مفتاح، فقط اتركيها كذا
        }

        payload = {
            "model": "clip-vit-base-patch32",
            "input": img_base64
        }

        response = requests.post(f"{JINA_URL}/encode", json=payload, headers=headers)

        if response.status_code != 200:
            return jsonify({"error": "API error", "details": response.text}), 500

        # قراءة الـ embedding من Jina
        result = response.json()
        embedding = result["data"][0]["embedding"]

        # هنا خذي embedding وقارنيه مع صورك داخل الملفات أو قاعدة البيانات
        # مؤقتًا نرجع الـ embedding فقط
        return jsonify({
            "message": "Success",
            "embedding_length": len(embedding),
            "embedding": embedding[:10]  # أول 10 عناصر فقط
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "Server is running without CLIP!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
