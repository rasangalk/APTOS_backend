from flask import Flask, request, jsonify
import numpy as np
from service.image_service import predict_disease
import cv2
from flask_cors import CORS
import base64
import os


APP_NAME = 'APTOS'

app = Flask(APP_NAME)
CORS(app)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'hello':'Hi world!'})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image uploaded'})

    # # Load and preprocess the image received from the Flask API
    # file = request.files['image']
    # image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    # prediction = predict_disease(image)

    # return jsonify({'predicted_class': prediction})

    try:
        data = request.json
        if 'image' in data:
            image_data = data['image']
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)

            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg')
            with open(filename, 'wb') as f:
                f.write(image_bytes)
            image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
            prediction = predict_disease(image)
            return jsonify({'predicted_class': prediction})
        else:
            return jsonify({'error': 'No image provided.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if APP_NAME == 'APTOS':
    app.run(debug=False, port=8081, host="0.0.0.0")

