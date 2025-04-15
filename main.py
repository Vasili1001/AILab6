import os
import time
import xml.etree.ElementTree as ET
import subprocess
import torch
import cv2
from collections import Counter

CLASSES = ['apple', 'banana', 'orange']

# 1. Установка YOLOv5 и зависимостей
def setup():
    if not os.path.exists("yolov5"):
        os.system("git clone https://github.com/ultralytics/yolov5")
    os.system("pip install -r yolov5/requirements.txt")
    os.system("pip install opencv-python xmltodict pyyaml")

# 2. Конвертация аннотаций
def convert_annotation(image_id, in_dir, out_dir):
    in_file = open(os.path.join(in_dir, image_id + ".xml"))
    tree = ET.parse(in_file)
    root = tree.getroot()
    out_file = open(os.path.join(out_dir, image_id + ".txt"), "w")
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    if w == 0 or h == 0:
        print(f"[Пропущено] {image_id}.xml: width or height is 0")
        return
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find("bndbox")
        b = (int(xmlbox.find("xmin").text), int(xmlbox.find("ymin").text),
             int(xmlbox.find("xmax").text), int(xmlbox.find("ymax").text))
        x = (b[0] + b[2]) / 2.0 / w
        y = (b[1] + b[3]) / 2.0 / h
        bw = (b[2] - b[0]) / w
        bh = (b[3] - b[1]) / h
        out_file.write(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

def convert_all():
    for split in ["train", "test"]:
        in_dir = f"data/{split}"
        out_dir = f"labels/{split}"
        os.makedirs(out_dir, exist_ok=True)
        for file in os.listdir(in_dir):
            if file.endswith(".xml"):
                image_id = file[:-4]
                convert_annotation(image_id, in_dir, out_dir)

# 3. Создание data.yaml
def create_yaml():
    with open("data/dataset.yaml", "w") as f:
        f.write("train: ../data/train\n")
        f.write("val: ../data/test\n")
        f.write("nc: 3\n")
        f.write("names: ['apple', 'banana', 'orange']\n")

# 4. Обучение модели
def train_model():
    from yolov5.train import run
    run(imgsz=640,
        batch=8,
        epochs=5,
        data='data/dataset.yaml',
        weights='yolov5s.pt',
        project='fruit_model',
        name='exp',
        exist_ok=True)

# 5. Детекция и размытие
def detect_and_blur(image_path, output_path):
    model = torch.hub.load('yolov5', 'custom', path='fruit_model/exp/weights/best.pt', source='local')
    model.conf = 0.4
    img = cv2.imread(image_path)
    results = model(image_path)
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        roi = img[y1:y2, x1:x2]
        roi_blur = cv2.GaussianBlur(roi, (51, 51), 0)
        img[y1:y2, x1:x2] = roi_blur
    cv2.imwrite(output_path, img)

# 6. Подсчёт объектов
def count_objects(image_path):
    model = torch.hub.load('yolov5', 'custom', path='fruit_model/exp/weights/best.pt', source='local')
    model.conf = 0.4
    results = model(image_path)
    classes = results.names
    detected = [classes[int(cls)] for cls in results.pred[0][:, -1]]
    counts = Counter(detected)
    print("Обнаруженные объекты:")
    for fruit, num in counts.items():
        print(f"  {fruit}: {num}")

# ⏱ Запуск всех шагов
if __name__ == "__main__":
    print("[1/6] Установка окружения...")
    setup()

    print("[2/6] Конвертация аннотаций...")
    convert_all()

    print("[3/6] Создание YAML-файла...")
    create_yaml()

    print("[4/6] Обучение YOLOv5...")
    train_model()

    test_image = "data/test/apple_77.jpg"
    print("[5/6] Детекция и размытие объектов...")
    detect_and_blur(test_image, "blurred_output.jpg")

    print("[6/6] Подсчёт объектов...")
    count_objects(test_image)

    print("\n✅ Всё готово. Посмотри результат: blurred_output.jpg")
