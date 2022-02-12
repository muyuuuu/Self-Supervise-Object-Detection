import json, random, os
from PIL import Image
import numpy as np

random.seed(100)

if not os.path.exists("data"):
    os.mkdir("data")

with open("../train.json", "r") as f:
    data = json.load(f)

label_to_image = {}
label_to_width = {}
label_to_height = {}
for item in data["images"]:
    label_to_image[item["id"]] = item["file_name"]
    label_to_width[item["id"]] = item["width"]
    label_to_height[item["id"]] = item["height"]

pos = 0
neg = 0
size = 224
root = "/home/muyu/Downloads/Flow/FloW_IMG/training/images/"

labels = {}
files = []
cnt = 0


def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    a = len(bounding_boxes)
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    if a == len(picked_boxes):
        return True
    return False


for image_id in range(1, 1200 + 1):

    boxes = []
    for item in data["annotations"]:
        if item["image_id"] == image_id:
            boxes.append(item)

    width = label_to_width[image_id]
    height = label_to_height[image_id]
    image = root + label_to_image[image_id]
    im = Image.open(image)

    bounding_boxes = []
    for i in boxes:
        b = i["bbox"]
        b[2] += b[0]
        b[3] += b[1]
        bounding_boxes.append(b)

    score = [1.0 for i in range(len(bounding_boxes) + 1)]

    box = random.choice(boxes)
    x, y, w, h = box["bbox"]
    w -= x
    h -= y

    scale = random.uniform(0.8, 1.2)
    new_size = int(size * scale)

    # 切负样本
    tmp = 0
    while tmp <= 10:
        x1 = random.randint(0, width - new_size - 1)
        y1 = random.randint(0, height - new_size - 1)
        bounding_boxes.append([x1, y1, x1 + new_size, y1 + new_size])
        a = nms(bounding_boxes, score, 0.005)
        if a is True:
            im2 = im.crop((x1, y1, x1 + new_size, y1 + new_size))
            im2.save("data/neg_{}.png".format(cnt), "PNG")
            cnt += 1
            neg += 1
            break
        bounding_boxes.pop()
        tmp += 1

    x0 = x - (new_size - w) // 2
    y0 = y - (new_size - h) // 2

    if w >= new_size:
        x0 = x
    if h >= new_size:
        y0 = y

    if x0 < 0:
        x0 = 0
    elif x0 + new_size > width:
        x0 = width - new_size

    if y0 < 0:
        y0 = 0
    elif y0 + new_size > height:
        y0 = height - new_size

    # 切正样本
    im1 = im.crop((x0, y0, x0 + new_size, y0 + new_size))
    im1.save("data/pos_{}.png".format(cnt), "PNG")
    pos += 1
    cnt += 1

print(cnt, pos, neg)
