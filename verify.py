from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import base64
import zipfile

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

def preprocess_image(img):
    img = cv2.resize(img, (64, 64))  
    img = img.astype('float32') / 255.0  # Normalize the image
    return img

def get_embedding(model, img):
    img = np.expand_dims(img, axis=0)
    return model.predict(img)

def verify_images(model, img1, img2):
    embedding1 = get_embedding(model, img1)
    embedding2 = get_embedding(model, img2)
    distance = np.linalg.norm(embedding1 - embedding2)
    threshold = 0.5

    print(f"Distance between embeddings: {distance}")

    if distance < threshold:
        return True
    else:
        return False

def load_compressed_model(json_path, zip_path):
    # Load the model architecture
    with open(json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)

    # Extract the weights file from the zip archive
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

    # Load the weights into the model
    weights_path = zip_ref.namelist()[0]
    model.load_weights(weights_path)

    os.remove(weights_path)

    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    model_json_path = 'cnn_face_verification_architecture.json'
    weights_zip_path = 'cnn_face_verification_weights.weights.zip'

    # Load the trained model
    model = load_compressed_model(model_json_path, weights_zip_path)

    # Decode and preprocess the first image from the camera
    data = request.get_json()
    img1_base64 = data['img1']
    img1_decoded = np.frombuffer(base64.b64decode(img1_base64.split(',')[1]), np.uint8)
    img1 = cv2.imdecode(img1_decoded, cv2.IMREAD_COLOR)
    img1_processed = preprocess_image(img1)

    # Decode and preprocess the second image from the camera
    img2_base64 = data['img2']
    img2_decoded = np.frombuffer(base64.b64decode(img2_base64.split(',')[1]), np.uint8)
    img2 = cv2.imdecode(img2_decoded, cv2.IMREAD_COLOR)
    img2_processed = preprocess_image(img2)

    # Verify if the two images are of the same person
    is_same_person = verify_images(model, img1_processed, img2_processed)

    if is_same_person:
        message = "The images are of the same person."
    else:
        message = "The images are of different people."

    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)
