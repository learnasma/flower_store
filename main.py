from flask import Flask, request, jsonify
import torch
import clip
from PIL import Image

app = Flask(__name__)

# تحميل النموذج وسمات الصور
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
data = torch.load("image_features.pt")
image_names = data["names"]
image_features = data["features"].to(device)

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

if __name__ == "__main__":
    # تشغيل السيرفر على الشبكة الداخلية
    app.run(host="0.0.0.0", port=5000, debug=True)
