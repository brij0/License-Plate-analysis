from ultralytics import YOLO
import torch

def main():
    model = YOLO("yolov8n.yaml")
    

    # Train the model
    results = model.train(data="Python files\Config.yaml", epochs=3)

if __name__ == '__main__':
    main()



success = model.export(format="onnx")  # export the model to ONNX format


#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124