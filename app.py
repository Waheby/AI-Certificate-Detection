#GENERAL API REQUEST IMPORTS
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve

# IMAGE CLASSIFICATION LOGIC =================================================================
from tensorflow.keras.models import load_model # type: ignore
import cv2
import tensorflow as tf
import os
import numpy as np
import urllib


# Create a Flask application instance
app = Flask(__name__)
# Enable CORS for all routes, allowing requests from any origin
CORS(app,resources={r"/*":{"origins":"*"}})

#   Define a route for HTTP request
@app.route('/validate', methods=['POST'])
def verify_certificate_image():
    try:
        # Receive Data from Frontend API Request
        data = request.get_json()

        req = urllib.request.urlopen(data)
        
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, 3)

        resize = tf.image.resize(img, (256,256))

        new_model = load_model(os.path.join('models', 'imageclassifier.h5'), compile=False)

        yhat = new_model.predict(np.expand_dims(resize/255, 0))

        if yhat > 0.5:
            print(f'NOT CERT AT ALL')
            result = False
        else:
            print(f'COULD BE CERT')
            result = True
        
        return jsonify({'Certificate Validity': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# OPTICAL CHARACTER RECOGNITION (OCR) LOGIC  =================================================================
import matplotlib.pyplot as plt
import keras_ocr

def detect_certificate_text():
    try:
        # Receive Data from Frontend API Request
        data = request.get_json()

        result = False

        pipeline = keras_ocr.pipeline.Pipeline()

        images = [
            data,
        ]

        words_found = [""]
        certificate_related_words = ["certificate", "certification", "university", "college", "institue", "degree", "graduate", "graduation" "coursework", "course", "certify", "association", "education", "organization", "program", "school", "award", "research", "appreciation", "presents", "present", "presented", "ministry", "certificat", "cisco", "google", "academy", "academic", "completed", "completion", "complete", "finished", "achievement", "pass", "passed", "programme", "program", "bachelor", "accreditation", "docotorate", "diploma", "training", "certifies", "certified", "honor", "honors", "department", "project", "congratulates", "congratulation", "professor", "digital", "requirements", "requirement", "doctor", "science", "president", "dean", "director", "ceo", "participation", "skill", "skills", "employment", "centre", "learn", "learning"]

        prediction = pipeline.recognize(images)

        for text, box in prediction[0]:
            words_found.append(text)

        print(words_found)

        for word in words_found:
            if word in certificate_related_words:
                print("a word exist")
                print(word)
                result = True
                break
            else:
                continue
        jsonify({'Certificate Words Found': result})

    except Exception as e:
        return jsonify({'error': str(e)})



# OBJECT DETECTION LOGIC =================================================================
from ultralytics import YOLO

def detect_certificate_objects():
    try:
        # Receive Data from Frontend API Request
        data = request.get_json()

        prediction = False

        # Load my newly created model
        model = YOLO("models/best.pt") 

        # Run batched inference on a list of images
        results = model([data])

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            
            if boxes:
                print("Valid Cert")
                prediction = True
            else:
                print("Invalid Cert")
                prediction = False
        
        return jsonify({'Object Detected': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)
    