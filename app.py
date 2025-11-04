from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

map_ = {0: "cat", 1: "dog", 2: "wild"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("animal_predict_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return map_[pred], probs[0][pred].item()

# 메인 페이지
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
            label, prob = predict_image(img_path)
            return render_template("index.html", filename=file.filename, label=label, prob=prob)
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return f"/{UPLOAD_FOLDER}/{filename}"

if __name__ == "__main__":
    app.run(debug=True)
