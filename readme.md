Install the required packages with the following command:

```bash
pip install --force-reinstall --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install -U Flask jinja2 imageio-ffmpeg opencv-python av
pip install -U watchdog
```

Generate a dummy video with the following command:

```bash
python create_dummy_video.py
```

Then run the flask web server with the following command:

```bash
python webapp.py
```

Click on the classify button and see the result.