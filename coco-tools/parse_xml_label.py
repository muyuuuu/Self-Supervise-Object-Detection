from xml.dom.minidom import parse
import os

label = set()
root = [
    "/home/muyu/Downloads/Flow/FloW_IMG/training/annotations/",
    "/home/muyu/Downloads/Flow/FloW_IMG/test/annotations/",
]
for idx, r in enumerate(root):
    label_file = "sample/test_xml.txt"
    if idx == 0:
        label_file = "sample/train_xml.txt"
    print(r, label_file)
    for file in os.listdir(r):
        file = os.path.join(r, file)
        with open(label_file, "a+") as f:
            f.write(file)
            f.write("\n")
        dom = parse(file)
        data = dom.documentElement
        a = data.getElementsByTagName("object")
        # a = a.getAttribute("object")
        for obj in a:
            name = obj.getElementsByTagName("name")[0].childNodes[0].nodeValue
            label.add(name)

print(label)
