# makes small tiles from the big PKU boards so YOLO can train on them
# run from the PCBD folder: py pku_to_yolo.py

import os
import random
import xml.etree.ElementTree as ET
from PIL import Image

src = "Data_app_test/PKU-Market-PCB"
out = "Data_app_test/PKU_tiles"
tile = 320  # tile size, same as in the app

# my model's class numbers, missing_hole is new so it gets number 9 (after BMFO)
classes = {"short": 0, "spur": 1, "spurious_copper": 2, "open_circuit": 3, "mouse_bite": 4, "missing_hole": 9}

# board 11 and 12 are for testing
val_boards = ["11", "12"]

random.seed(0)

os.makedirs(out + "/images/train", exist_ok=True)
os.makedirs(out + "/images/val", exist_ok=True)
os.makedirs(out + "/labels/train", exist_ok=True)
os.makedirs(out + "/labels/val", exist_ok=True)

count = 0

for folder in os.listdir(src):
    if not os.path.isdir(src + "/" + folder) or folder.startswith("."):
        continue

    for file in os.listdir(src + "/" + folder):
        if not file.endswith(".xml"):
            continue

        # read the xml
        root = ET.parse(src + "/" + folder + "/" + file).getroot()
        board = root.find("filename").text.split("_")[0]

        boxes = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name in classes:
                b = obj.find("bndbox")
                xmin = int(b.find("xmin").text)
                ymin = int(b.find("ymin").text)
                xmax = int(b.find("xmax").text)
                ymax = int(b.find("ymax").text)
                boxes.append([classes[name], xmin, ymin, xmax, ymax])

        if len(boxes) == 0:
            continue

        if board in val_boards:
            split = "val"
        else:
            split = "train"

        name = file.replace("_annotation.xml", "")
        img = Image.open(src + "/" + folder + "/" + name + ".jpg")
        img_w, img_h = img.size

        # make the tile positions: 3 tiles around every defect + 1 empty tile
        tiles = []
        for box in boxes:
            for k in range(3):
                # middle of the defect, moved a bit randomly so its not always in the center
                mid_x = (box[1] + box[3]) // 2 + random.randint(-80, 80)
                mid_y = (box[2] + box[4]) // 2 + random.randint(-80, 80)
                x = mid_x - tile // 2
                y = mid_y - tile // 2
                # dont go outside the board
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if x > img_w - tile:
                    x = img_w - tile
                if y > img_h - tile:
                    y = img_h - tile
                tiles.append([x, y])

        # empty tile: try random spots until one has no defect in it
        for t in range(50):
            x = random.randint(0, img_w - tile)
            y = random.randint(0, img_h - tile)
            empty = True
            for box in boxes:
                if box[1] < x + tile and box[3] > x and box[2] < y + tile and box[4] > y:
                    empty = False
            if empty:
                tiles.append([x, y])
                break

        # cut every tile and write its labels
        for i in range(len(tiles)):
            x = tiles[i][0]
            y = tiles[i][1]
            lines = []

            for box in boxes:
                # cut the box at the tile edges
                x1 = max(box[1], x)
                y1 = max(box[2], y)
                x2 = min(box[3], x + tile)
                y2 = min(box[4], y + tile)
                if x2 <= x1 or y2 <= y1:
                    continue  # box is not in this tile

                # only keep it if at least half of the box is inside
                part_inside = (x2 - x1) * (y2 - y1) / ((box[3] - box[1]) * (box[4] - box[2]))
                if part_inside < 0.5:
                    continue

                # tile coordinates, then 0-1 like YOLO wants
                x_center = ((x1 + x2) / 2 - x) / tile
                y_center = ((y1 + y2) / 2 - y) / tile
                w = (x2 - x1) / tile
                h = (y2 - y1) / tile
                lines.append(str(box[0]) + " " + str(round(x_center, 6)) + " " + str(round(y_center, 6))
                             + " " + str(round(w, 6)) + " " + str(round(h, 6)))

            img.crop((x, y, x + tile, y + tile)).save(out + "/images/" + split + "/" + name + "_" + str(i) + ".jpg")
            f = open(out + "/labels/" + split + "/" + name + "_" + str(i) + ".txt", "w")
            f.write("\n".join(lines))
            f.close()
            count += 1

print("done,", count, "tiles")
print("train:", len(os.listdir(out + "/images/train")))
print("val:", len(os.listdir(out + "/images/val")))
