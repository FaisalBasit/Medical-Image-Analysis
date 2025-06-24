from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from huggingface_hub import hf_hub_download
import numpy as np
import os
import json
import asyncio
import markdown
from agent_helper import (
    get_treatment_plan,
    get_condition_explanation,
    get_doctor_recommendation
)

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Brain Tumor Model from Hugging Face
brain_model_path = hf_hub_download(repo_id="faisal159/Brain-Tumor-Detection", filename="model.h5")
brain_model = load_model(brain_model_path)

# Load Lung Disease Model from Hugging Face
lung_model_path = hf_hub_download(repo_id="faisal159/Lung-Cancer-Detection", filename="lung_model.h5")
lung_model = load_model(lung_model_path)


# Load brain class label mappings
with open('models/class_indices.json') as f:
    brain_class_indices = json.load(f)
brain_index_to_label = {v: k for k, v in brain_class_indices.items()}

# Image sizes
IMAGE_SIZE_BRAIN = 128
IMAGE_SIZE_LUNG = 150

# Prediction for brain tumor
def predict_brain(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE_BRAIN, IMAGE_SIZE_BRAIN))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = brain_model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    label = brain_index_to_label[predicted_index]
    result = "No Tumor" if label == 'notumor' else f"Tumor: {label}"
    return result, confidence

# Prediction for lung image
def predict_lung(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE_LUNG, IMAGE_SIZE_LUNG))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = lung_model.predict(img_array)[0][0]
    predicted_class = "Normal" if prediction < 0.5 else "Pneumonia"
    confidence = (1 - prediction) if predicted_class == "Normal" else prediction
    return f"Lung Scan: {predicted_class}", confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        detection_type = request.form['detection_type']
        city = request.form.get('city', '')  # User input for city

        if file and detection_type:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Predict based on selected type
            if detection_type == 'brain':
                result, confidence = predict_brain(file_path)
            elif detection_type == 'lung':
                result, confidence = predict_lung(file_path)
            else:
                result, confidence = "Invalid detection type", 0

            # Run AI agents
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            condition_text = loop.run_until_complete(get_condition_explanation(result))
            treatment_plan_text = loop.run_until_complete(get_treatment_plan(result))
            doctor_text = loop.run_until_complete(get_doctor_recommendation(result, city))

            return render_template('index.html',
                                   result=result,
                                   confidence=f"{confidence * 100:.2f}%",
                                   condition_explained=markdown.markdown(condition_text),
                                   treatment_plan=markdown.markdown(treatment_plan_text),
                                   doctor_recommendations=markdown.markdown(doctor_text),
                                   file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
