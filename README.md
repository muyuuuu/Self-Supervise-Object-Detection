[简体中文](https://github.com/muyuuuu/Flow-Detection/blob/main/README-zh.md)

# YOLOX-S Object Detection

The dataset is the [FloW](http://www.orca-tech.cn/datasets/FloW/FloW-Img) dataset from Ouka, using `mmdetection` as a tool.

<p align="center">
    <img src="./sample/1.jpg" width="400">
    <img src="./sample/res.gif" width="400">
</p>

- `coco-tools` below is the markup from `VOC` format to `json` format
- `mmdetction` is the configuration file, results and logs involved, there is no need to submit the whole `mmdetection` project
- `crop-image` is the cut image, used to pre-train the backbone

# Optimize

Idea: Considering the arithmetic power and performance, choose single-stage `YOLOX-s` as `baseline`. **refuse to use** the commonly used trick for competitions, including but not limited to: model fusion, large-scale backbone such as swin, cascade faster rcnn, various strategy combinations.

The very first idea is to use self-supervision to improve detection performance, after all, no labeling is required. But self-supervision + object detection is not elegant enough for the current paper, so I decided to give up and leave it as a major direction for the future.

## Strategy 1

But I found that the precision and recall of `baseline` is not very good, so is there a simple way to improve it? What I can think of is `backbone` can not use COCO pre-training experience, but pre-train `backbone` for this problem, and this `backbone` can effectively identify the foreground and background.

Reading the source code reveals that it is not difficult to implement a pre-trained `backbone`, under `mmdetection/tools`.

- `center_loss.py`, for the small recall improvement, read the source code of `YOLOX` and analyzed the reason, thought it was because the distinction of the front background features extracted by `backbone` was not obvious, which led to the possibility that the back `neck` and `head` thought the background features were foreground and the foreground features were background. So we use `center loss` to increase the differentiation of the representation. The accuracy of distinguishing the foreground from the background is 96.67%, +5.3% mAP, +3.2% mAR. Ablation experiments show that center loss is better than cross entropy loss alone.
- `pretrain.py`, loaded pretrained `backbone` at detection

## Strategy 2

But strategy 1 also brings a problem that YOLOX-tiny does not improve significantly using this strategy, and YOLOX-tiny outperforms YOLOX-S 3.2% mAP in detection.

After reading the source code, we found that YOLOX's SimOTA mechanism has some loopholes when assigning samples to small targets, see the link on the right side of the repository for a detailed analysis. In short, due to the small target, the selected positive sample and the real target do not intersect, the loss of cls and obj is fine, but the loss of reg is 0, which is unreasonable, and the effect of using CIoU Loss correction is also very obvious.