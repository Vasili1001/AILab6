import os
import xml.etree.ElementTree as ET

CLASSES = ['apple', 'banana', 'orange']  # Классы из датасета

def convert_annotation(image_id, in_dir, out_dir):
    in_file = open(os.path.join(in_dir, image_id + ".xml"))
    tree = ET.parse(in_file)
    root = tree.getroot()

    out_file = open(os.path.join(out_dir, image_id + ".txt"), "w")
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find("bndbox")
        b = (int(xmlbox.find("xmin").text), int(xmlbox.find("ymin").text),
             int(xmlbox.find("xmax").text), int(xmlbox.find("ymax").text))
        # YOLO формат
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

if __name__ == "__main__":
    convert_all()
