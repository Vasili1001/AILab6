import os
os.system("git clone https://github.com/ultralytics/yolov5")
os.system("pip install -r yolov5/requirements.txt")

from yolov5.train import run

run(imgsz=640,
    batch=8,
    epochs=20,
    data='data/dataset.yaml',
    weights='yolov5s.pt',
    project='fruit_model',
    name='exp',
    exist_ok=True)
