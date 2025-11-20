from flask import Flask, request, jsonify
import torch
import clip
from PIL import Image
import os

app = Flask(__name__)

# =========================
# 1. تحميل نموذج CLIP تلقائيًا
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B/32"

print("Loading CLIP model...")

# تحميل النموذج - لو غير موجود بيتم تحميله تلقائيًا
model, preprocess = clip.load(MODEL_NAME, device=device)

print("Model loaded successfully!")

# =========================
# 2. تحميل خصائص (Features) الصور
# =========================

FEATURES_FILE = "image_features.pt"

if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError(f"{FEATURES_FILE} not found! You must upload it to GitHub.")

data = torch.load(FEATURES_FILE, map_location=device)
image_names = data["names"]
image_features = data["features"].to(device)

# =========================
# 3. API — البحث عن الصور المشابهة
# =========================

@app.route("/search", methods=["POST"])
def search():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    img_file = request.files["image"]
    img = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_feature = model.encode_image(img)
        query_feature /= query_feature.norm(dim=-1, keepdim=True)
    
    similarities = (image_features @ query_feature.T).squeeze(1)
    top_k = min(3, similarities.shape[0])
    top_probs, top_idxs = similarities.topk(top_k)
    
    results = [image_names[int(idx)] for idx in top_idxs]
    return jsonify({"results": results})

@app.route("/")
def home():
    return "Server Running!"

# =========================
# 4. تشغيل السيرفر
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
