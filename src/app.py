import numpy as np
import torch
from torchvision import transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
import model
import util

app = Flask(__name__)
model = model.CNN()
util.load_model(model, 'model_b_32_e_10')

def apply_preprocessing(img):
    my_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((28, 28)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, ), (0.5, ))])
    img = Image.open(io.BytesIO(img))
    img = my_transforms(img)
    img = img.view(-1, 1, 28, 28).float()
    return img

def get_prediction(x):
    predictions = model.forward(x)
    return torch.argmax(predictions)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        x = apply_preprocessing(img_bytes)
        digit = get_prediction(x)
        return jsonify({'predicted_digit': digit.numpy().tolist()})
        
if __name__ == "__main__":
    app.run(port=8000, host="0.0.0.0")
