import torch
from collections import Counter

model = torch.hub.load('yolov5', 'custom', path='fruit_model/exp/weights/best.pt', source='local')
model.conf = 0.4

def count(image_path):
    results = model(image_path)
    classes = results.names
    detected = [classes[int(cls)] for cls in results.pred[0][:, -1]]
    counts = Counter(detected)
    for fruit, num in counts.items():
        print(f"{fruit}: {num} found")

if __name__ == "__main__":
    count("data/test/apple_77.jpg")
