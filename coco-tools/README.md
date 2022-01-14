- `parse_xml_label.py` 生成文件列表和标签列表
- `voc2coco.py` 转换文件格式

```shell
python voc2coco.py \
             --ann_paths_list sample/test_xml.txt \
             --labels sample/labels.txt \
             --output test.json
```