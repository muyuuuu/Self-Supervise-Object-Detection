数据集为欧卡的 [FloW](http://www.orca-tech.cn/datasets/FloW/FloW-Img) 数据集，使用了 `mmdetection` 作为工具。

<p align="center">
    <img src="./sample/1.jpg" width="400">
    <img src="./sample/res.gif" width="400">
</p>

- `coco-tools` 下面是将 `VOC` 格式的标注转 `json` 格式的标注
- `mmdetction` 是涉及的配置文件，把整个 `mmdetection` 提交上来没意思
- `crop-image` 是切割图片，对应到自己的路径中

思路：考虑到算力和性能，选用单阶段 `YOLOX-s` 作为 `baseline`。这里表示拒绝使用竞赛常用的上分 trick，包括但不限于：模型融合、cascade faster rcnn、各种策略组合。

但是我发现 `baseline` 的精度和召回率不是很好，那么有没有一种简单的提升方法呢？我看了 `YOLOX` 的论文，数据增强、采样、模型结构都无可挑剔了，我能想到的就是 `backbone` 能不能不使用 COCO 预训练的经验，而是针对这个问题预训练 `backbone`，而这个 `backbone` 能有效识别前景和背景。

于是开始读源码并按照自己的想法进行修改，`mmdetection/tools` 下：

- `cls.py` 单纯的预训练 `backbone` 去区分前背景，精度 94.01%，+4.2% mAP, +1.9%mAR
- `cls_center.py`，针对召回率提升较小，读了 `YOLOX` 的源代码并分析了下原因，认为是 `backbone` 提取的前背景特征区分度不明显，导致后面的 `neck` 和 `head` 可能认为背景特征是前景，前景特征是背景。于是使用 `center loss` 增加表示的区分度。区分前背景的精度为 96.67%，+5.3% mAP, +3.2% mAR
- `pretrain.py`，检测时加载 `backbone`