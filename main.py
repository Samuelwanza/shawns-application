from flask import Flask, render_template, request, jsonify
import os
import requests
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# model = load_model('defects_model1.h5')  # Update with your model path
model_url='https://drive.google.com/uc?id=1X6ojdm6dzwiNRi1n5EbGjwaQwKTM9Etc'
model_path='magnetic_tile_defect_model.h5'
response = requests.get(model_url)


with open(model_path, 'wb') as f:
    f.write(response.content)
model = load_model(model_path)

def preprocess_image(image_path, label):
    image = Image.open(image_path)  # Use Pillow to open the image
    image = image.resize((299, 299))  # Resize to the desired size

    # Convert the grayscale image to RGB
    if image.mode == "L":
        image = image.convert("RGB")

    # Preprocess the image for VGG16
    image_array = np.array(image)
    image_array = image_array.astype(np.float32)
    image_array[:, :, 0] -= 103.939
    image_array[:, :, 1] -= 116.779
    image_array[:, :, 2] -= 123.68

    return image_array, label

@app.route('/')
def home():
    return 'index.html'

@app.route('/upload', methods=['POST'])
def upload_file():
    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file part'})

    file = request.files['file']
    print(file)

    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    print(file_path)

    image, _ = preprocess_image(file_path, label=None)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)


    # Return the prediction as JSON
    modi_prediction=float(prediction[0][0])
    if modi_prediction>0.5:
        return jsonify({'result':'Defective'})
    else:
        return jsonify({'result':'Non-Defective'})
    

if __name__ == '__main__':
    app.run(port=os.getenv('PORT', default=5001))

