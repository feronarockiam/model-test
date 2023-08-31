from flask import Flask, request, jsonify
import numpy as np
from ultralytics import YOLO
import dropbox
import os

app = Flask(__name__)

# Replace with your Dropbox access token
DROPBOX_ACCESS_TOKEN = 'sl.BlJ-DI7j26UDJb-Qd_F5pyMiMtSZdNTqLm9FxuFf5z2-H3pTxMfBeIW-sHDoUl-sbI7xpFdly9s_dfxcY56CleBQTea1pNHXN0cy8IftSLBMopf9ig9k5Am_uHuPCH6fu-JgY7F12wqF'

def download_best_pt():
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    dropbox_file_path = 'https://www.dropbox.com/scl/fi/fbb9og5o0cd5ar54ffgbu/best.pt?rlkey=xt3q2tbbjg5m15xuxa3t9wdki&dl=0'  # Replace with the actual Dropbox file path
    local_file_path = './best.pt'

    with open(local_file_path, 'wb') as f:
        metadata, res = dbx.files_download(dropbox_file_path)
        f.write(res.content)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Download the best.pt file from Dropbox
        download_best_pt()

        # Load a model
        model = YOLO('./best.pt')  # pretrained YOLOv8n model

        # Run batched inference on a list of images
        results = model(['./images/DJI_0248.jpg'])  # return a list of Results objects
        print(results)

        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
