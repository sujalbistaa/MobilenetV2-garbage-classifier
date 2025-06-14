from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

# Modify model loading for deployment
MODEL_PATH = "C:\\Users\\acer\\Desktop\\CV\\mobilenetv2_best.keras"
model = load_model(MODEL_PATH)
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        try:
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            predictions = model.predict(img_array)
            predicted_class = CLASSES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            
            os.remove(filepath)
            
            return jsonify({
                'class': predicted_class,
                'confidence': f'{confidence:.2%}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
