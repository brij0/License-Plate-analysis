from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.yaml")
    # Train the model
    results = model.train(data="Python files\Config.yaml", epochs=10)

if __name__ == '__main__':
    main()


#pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
