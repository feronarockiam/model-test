from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        

        # Load a model
        model = YOLO('./best.pt')  # pretrained YOLOv8n model

        # Run batched inference on a list of images
        results = model(['./images/DJI_0248.jpg'])  # return a list of Results objects
        print(results)


        return jsonify(results),200
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
