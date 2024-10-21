from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.yaml")
    results = model.train(data="Python files\Config.yaml", epochs=10)

if __name__ == '__main__':
    main()
