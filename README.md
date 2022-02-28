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

## Strategy 1, Center Loss to pretrain backbone

But I found that the precision and recall of `baseline` is not very good, so is there a simple way to improve it? What I can think of is `backbone` can not use COCO pre-training experience, but pre-train `backbone` for this problem, and this `backbone` can effectively identify the foreground and background.

Reading the source code reveals that it is not difficult to implement a pre-trained `backbone`, under `mmdetection/tools`.

- `center_loss.py`, for the small recall improvement, read the source code of `YOLOX` and analyzed the reason, thought it was because the distinction of the front background features extracted by `backbone` was not obvious, which led to the possibility that the back `neck` and `head` thought the background features were foreground and the foreground features were background. So we use `center loss` to increase the differentiation of the representation. The accuracy of distinguishing the foreground from the background is 96.67%, +5.3% mAP, +3.2% mAR. Ablation experiments show that center loss is better than cross entropy loss alone.
- `pretrain.py`, loaded pretrained `backbone` at detection

## Strategy 2, CIoU Loss to fix SimOTA vulnerability

But strategy 1 also brings a problem that YOLOX-tiny does not improve significantly using this strategy, and YOLOX-tiny outperforms YOLOX-S 3.2% mAP in detection.

After reading the source code, we found that YOLOX's SimOTA mechanism has some loopholes when assigning samples to small targets, see the link on the right side of the repository for a detailed analysis. In short, due to the small target, the selected positive sample and the real target do not intersect, the loss of cls and obj is fine, but the loss of reg is 0, which is unreasonable, and the effect of using CIoU Loss correction is also very obvious.

## Strategy 3, Self Supervised Training

In the real world not all data is labeled. So how to make good use of unlabeled data? In the context of papers I've read before.

- Microsoft has published a SOTA-related article in ICCV 2021 about semi-supervised target detection,  but it is complicated to adjust the parameters, and the model capacity has to be doubled which is not friendly to non-RMB players.
- In the field of self-supervised object detection, DetCo is based on the Moco improvement whose papers and code I have read and found to be unfriendly to non-RMB players, and the Facebook AI Institute out of Moco and Simsiam idea is relatively novel and simple, but not easy to accept.
- The baseline chosen by self-EMD is BYOL, and the derivation of the formula in it is relatively nice, but the structure of BYOL is dissuasive.

In summary, is there a simple self-supervised training method that can be used for target detection in specific scenarios? Inspired by self-EMD, I have done the following simple work.

<p align="center">
    <img src="./sample/ssl.jpg" width="600">
</p>

- Cut out several patches in the image, with blue in the middle as anchor, pink as positive samples, and purple as negative samples
- Using cosine distance as the loss function, the representation of anchor and positive samples should be close to each other, and the representation of anchor and negative samples should be far from each other
- Considering that the target detection is influenced by spatial information, the patch of positive samples must be next to the anchor.

The experimental results show that this pre-training approach is better than the labeled training approach. Here I just give my thoughts: 

- For the labeled training method, the background and target are obtained by cutting the image, then the network only knows the background and the target, throwing a complete image, and the network is only interested in the target region; 
- if it is self-supervised training, then all the network knows is the distribution of the data, or what the image should look like, and it is not particularly interested in a certain region; but when the detection program starts training and needs to be interested in certain regions, the network knows which regions it needs to be interested in, which regions are similar to the regions of interest, and which regions are not, so that it can better locate the target.