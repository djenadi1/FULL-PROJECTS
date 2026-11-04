if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("yolo26m-seg.pt")
    model.train(data ='../data/data.yaml', epochs =100, time =2.7, patience =100, batch= -1, imgsz = 1280, workers =6, name ='v2_1280_m')

