import base64, io
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file
from PIL import Image
from torchvision import transforms
from model import CNN

app = Flask(__name__)

model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def preprocess(img):
    """Gjør tegningen lik et MNIST-bilde: hvitt tall, sentrert, ca. 20×20 i en 28×28-ramme."""
    img = img.convert("L")
    bbox = img.getbbox()                  # område som inneholder tegning
    if bbox is None:
        return None
    img = img.crop(bbox)
    w, h = img.size
    scale = 20 / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)  # svart bakgrunn
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    return transform(canvas).unsqueeze(0)

@app.get("/")
def index():
    return send_file("index.html")

@app.post("/predict")
def predict():
    data_url = request.json["image"]      # "data:image/png;base64,...."
    img = Image.open(io.BytesIO(base64.b64decode(data_url.split(",")[1])))
    x = preprocess(img)
    if x is None:
        return jsonify(error="Tom tegning"), 400
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    return jsonify(digit=probs.argmax().item(), confidence=probs.max().item())
