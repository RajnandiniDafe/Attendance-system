from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')

def main():
    model.train(data='Dataset/SplitData/dataOffline.yaml', epochs=250)

if __name__ == '__main__':
    main()


