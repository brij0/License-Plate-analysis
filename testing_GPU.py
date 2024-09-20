import torch
from ultralytics import YOLO

# Check if GPU is available


if torch.cuda.is_available():
    print("GPU is available. Running on GPU.")
else:
    print("GPU is not available. Running on CPU.")

# # Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # You can change this to the appropriate model

# # Run inference (GPU will be used automatically if available)
# # results = model('path_to_your_image_or_video')

# # Display results
# print("Running")

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
