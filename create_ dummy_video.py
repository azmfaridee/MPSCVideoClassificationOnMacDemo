import cv2
import numpy as np

# Define video parameters
width = 320
height = 240
fps = 16
duration = 1  # in seconds

# Create video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dummy_video.mp4', fourcc, fps, (width, height))

# Generate and write frames to the video file
for _ in range(fps * duration):
    # Generate random frame
    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Write frame to video
    out.write(frame)

# Release video writer object
out.release()

print("Dummy video file generated successfully.")
