from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained YOLO model
model = YOLO('best.pt')

# Define class_dict
class_dict = {0: 'Badminton', 1: 'Cricket', 2: 'Karate', 3: 'Soccer', 4: 'Swimming', 5: 'Tennis', 6: 'Wrestling'}

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Make prediction
        results = model.predict(source=filepath, device='cuda')
        predicted_idx = results[0].probs.data.argmax().item()
        predicted_class = class_dict.get(predicted_idx, "Unknown Class")

        return jsonify({'predicted_class': predicted_class})

    return jsonify({'error': 'File upload failed'})

if __name__ == '__main__':
    app.run(debug=True)
