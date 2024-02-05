from torchvision.transforms import functional as F
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request
from PIL import Image
from torchvision.io import read_video
from torchvision.models.video import r2plus1d_18

use_mps = False

app = Flask(__name__)

# Load pre-trained R2Plus1D model
model = r2plus1d_18(pretrained=True)
if use_mps:
    model = model.to('mps')
model.eval()


# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for dummy video classification


@app.route('/classify_dummy', methods=['GET'])
def classify_dummy_video():
    dummy_video_path = 'dummy_video.mp4'

    if os.path.exists(dummy_video_path):
        # Read and preprocess video
        video_tensor, audio_tensor, info = read_video(
            dummy_video_path, pts_unit='sec')

        # shape of video_tensor: (T, H, W, C)
        video_tensor = video_tensor.float() / 255.0

        video_tensor = video_tensor.permute(3, 0, 1, 2)
        video_tensor = video_tensor.unsqueeze(0)

        if use_mps:
            video_tensor = video_tensor.to('mps')

        with torch.no_grad():
            outputs = model(video_tensor)

        predicted_class = torch.argmax(outputs, dim=1).item()

        return f"Predicted class for dummy_video.mp4: {predicted_class}"
    else:
        return "Dummy video file not found."


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
