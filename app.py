from flask import Flask, render_template, request, url_for, redirect
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


# 메인 페이지 (업로드 + 결과)
@app.route("/", methods=["GET", "POST"])
def index():
    filename = None
    label = None
    prob = None
    share_url = None
    show_form = True  # 기본: 업로드 폼 보이기

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filename = file.filename
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            # 모델 예측
            label, prob = predict_image(img_path)

            # 이 업로드 결과를 공유하는 링크 (폼 숨기고 결과만 보이는 페이지)
            share_url = url_for(
                "share",
                filename=filename,
                label=label,
                prob=prob,
                _external=True
            )

            return render_template(
                "index.html",
                filename=filename,
                label=label,
                prob=prob,
                share_url=share_url,
                show_form=True,   # 업로드 직후에는 폼도 그대로 보이게
            )

    # GET / : 첫 진입 → 폼만 보이게
    return render_template("index.html", show_form=True)


# 공유용 결과 페이지 (폼 없이 결과만)
@app.route("/share")
def share():
    filename = request.args.get("filename")
    label = request.args.get("label")
    prob_str = request.args.get("prob")

    if not (filename and label and prob_str):
        # 파라미터 이상하면 메인으로 돌려보내기
        return redirect(url_for("index"))

    prob = float(prob_str)

    # 자기 자신을 가리키는 공유 링크 (링크 안에서 또 복사해도 같은 URL)
    share_url = url_for(
        "share",
        filename=filename,
        label=label,
        prob=prob,
        _external=True
    )

    return render_template(
        "index.html",
        filename=filename,
        label=label,
        prob=prob,
        share_url=share_url,
        show_form=False,  # 🔥 공유 페이지에서는 업로드 폼 안 보이게
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return f"/{UPLOAD_FOLDER}/{filename}"


if __name__ == "__main__":
    app.run(debug=True)
