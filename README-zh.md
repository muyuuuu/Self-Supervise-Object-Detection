# YOLOX-S 目标检测

数据集为欧卡的 [FloW](http://www.orca-tech.cn/datasets/FloW/FloW-Img) 数据集，使用了 `mmdetection` 作为工具。

<p align="center">
    <img src="./sample/1.jpg" width="400">
    <img src="./sample/res.gif" width="400">
</p>

- `coco-tools` 下面是将 `VOC` 格式的标注转 `json` 格式的标注
- `mmdetction` 是涉及的配置文件、结果和日志，把整个 `mmdetection` 提交上来没意思
- `crop-image` 是切割图片，用于预训练backbone，对应到自己的路径中

# 调优

思路：考虑到算力和性能，选用单阶段 `YOLOX-s` 作为 `baseline`。**拒绝使用**竞赛常用的上分 trick，包括但不限于：模型融合、大规模 backbone 如 swin、cascade faster rcnn、各种策略组合。

最开始的想法是用自监督提升检测性能，毕竟不需要标签。但是自监督+目标检测目前论文提出来的方案我觉得不够优雅，遂决定放弃，留做日后的一个主攻方向。

## 策略一

但是我发现 `baseline` 的精度和召回率不是很好，那么有没有一种简单的提升方法呢？我能想到的就是 `backbone` 能不能不使用 COCO 预训练的经验，而是针对这个问题预训练 `backbone`，而这个 `backbone` 能有效识别前景和背景。

阅读源码发现实现预训练的 `backbone` 并不难，`mmdetection/tools` 下：

- `center_loss.py`，针对召回率提升较小，读了 `YOLOX` 的源代码并分析了下原因，认为是 `backbone` 提取的前背景特征区分度不明显，导致后面的 `neck` 和 `head` 可能认为背景特征是前景，前景特征是背景。于是使用 `center loss` 增加表示的区分度。区分前背景的精度为 96.67%，+5.3% mAP, +3.2% mAR。消融实验显示 center loss 好于单独的 cross entropy loss。
- `pretrain.py`，检测时加载 `backbone`

## 策略二

但是策略一的同时带来了一个问题，YOLOX-tiny 使用这种策略效果提升不明显，且 YOLOX-tiny 的检测效果优于 YOLOX-S 3.2%mAP，从源代码的角度分析了一下原因。

在一番阅读源码后，发现 YOLOX 的 SimOTA 机制在个小目标分配样本的时候存在一些漏洞，具体分析可以看仓库右侧的链接。简而言之，由于目标很小，选择的正样本和真实目标不相交，cls 和 obj 的损失没问题，但 reg 的损失为 0，这不合理，使用 CIoU Loss 修正，效果提升也很明显。