from flask import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import uuid

os.environ["HF_HOME"] = "/tmp/huggingface"

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device utilisé : {device}")

MODEL_NAME = "fatma29/blip-finetuned"

print("Chargement du modèle fine-tuné depuis Hugging Face...")

processor = BlipProcessor.from_pretrained(MODEL_NAME)

model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

model.to(device)
model.eval()

torch.set_num_threads(4)

print("Modèle fine-tuné prêt ")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@torch.no_grad()
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            max_length=50,
            num_beams=3
        )

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    except Exception as e:
        print("Erreur génération:", e)
        return "Erreur lors de la génération de la description"

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_path = None

    if request.method == "POST":

        if 'image' not in request.files:
            return "Aucune image envoyée"

        file = request.files['image']

        if file.filename == "":
            return "Nom fichier vide"

        if file and allowed_file(file.filename):

            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            caption = generate_caption(filepath)
            image_path = filepath

    return render_template("index.html", caption=caption, image_path=image_path)

if __name__ == "__main__":
    print("Flask démarré sur port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)