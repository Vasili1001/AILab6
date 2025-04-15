import cv2
import torch
from pathlib import Path

model = torch.hub.load('yolov5', 'custom', path='fruit_model/exp/weights/best.pt', source='local')
model.conf = 0.4

def blur_objects(image_path, output_path):
    img = cv2.imread(image_path)
    results = model(image_path)
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        roi = img[y1:y2, x1:x2]
        roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
        img[y1:y2, x1:x2] = roi_blur
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    blur_objects("data/test/apple_77.jpg", "blurred_result.jpg")
