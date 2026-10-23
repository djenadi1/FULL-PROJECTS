from ultralytics import YOLO

if __name__ == "__main__":
    YOLO("yolo26l.pt").train(
        data="combined.yaml", epochs=60, patience=15,
        imgsz=320, cache="ram", workers=4,
        project="runs", name="combined_l_320",
    )