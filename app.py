from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import sklearn
import pickle
import requests
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.disease import disease_dic
from utils.model import ResNet9
import os

# Define disease classes
disease_classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Define the model architecture
def load_model_from_pickle(pickle_path):
    """
    Load the model from a pickle file.
    :params: pickle_path (str)
    :return: model
    """
    with open(pickle_path, 'rb') as f:
        # Load the model architecture
        model = ResNet9(3, len(disease_classes))
        # Load the model state dictionary
        state_dict = pickle.load(f)
        model.load_state_dict(state_dict)
    return model

# Load the model
model_pickle_path = 'model.pkl'
disease_model = load_model_from_pickle(model_pickle_path)
disease_model.eval()

# Define the upload folder
UPLOAD_FOLDER = 'static'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label.
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

@app.route('/')
def home():
    title = 'Plant Disease Detection using Machine Learning'
    return render_template('disease.html', title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Plant Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg'))
            prediction = predict_image(img)

            # Retrieve the details for the detected disease
            disease_info = disease_dic.get(prediction, {
                'crop': 'Unknown',
                'disease': prediction,
                'cause': 'No information available.',
                'precautions': ['No information available.']
            })

            # Format the prediction as a string
            prediction_html = (
                f"<b>Crop</b>: {disease_info['crop']}<br/>"
                f"<b>Disease</b>: {disease_info['disease']}<br/>"
                f"<br/> <b>Cause of disease:</b> <br/><br/>{disease_info['cause']}<br/><br/>"
                f"<b>How to prevent/cure the disease:</b> <br/><br/>" +
                "".join([f"{i+1}. {measure}<br/>" for i, measure in enumerate(disease_info['precautions'])])
            )
            return render_template('disease-result.html', prediction=prediction_html, title=title)
        except Exception as e:
            return str(e)  # To see any exceptions that occur
    return render_template('disease.html', title=title)

if __name__ == "__main__":
    app.run(debug=True)
