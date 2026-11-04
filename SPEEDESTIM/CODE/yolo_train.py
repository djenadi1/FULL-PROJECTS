if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("yolo26s-seg.pt")  
    model.train(data ='../data/data.yaml',epochs =10 , batch= -1, imgsz = 980 )

