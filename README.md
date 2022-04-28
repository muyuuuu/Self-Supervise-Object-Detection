[简体中文](https://github.com/muyuuuu/Flow-Detection/blob/main/README-zh.md)

# Self-Supervised Object Detection

The dataset is the [FloW](http://www.orca-tech.cn/datasets/FloW/FloW-Img) dataset from Ouka, using `mmdetection` as a tool.

<p align="center">
    <img src="./sample/1.jpg" width="400">
    <img src="./sample/res.gif" width="400">
</p>

- `coco-tools` convert `VOC` format to `json` format
- `mmdetction` contains the configuration file, results and logs, there is no need to submit the whole `mmdetection` project
- `crop-image` is the crop image and used to pre-train the backbone, read_video reads video and generates unlabeled data for self-supervised training

Considering the computation power and performance, we choose the single stage `yolox-s` as the baseline. **Refuses to use tricks commonly used in** competitions, including but not limited to: model ensemble, large-scale backbone such as SWIN, Cascade Faster RCNN, and various strategy combinations.

### baseline

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.752
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.218
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.197
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.461
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.317
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.536
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.578
```

# Optimize

## Strategy 1, Center Loss pre-training backbone

Instead of using COCO's pre-training experience, we use backbone can be pretrained to recognize foreground and background.

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.781
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.492
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.587
```

<details><summary>Details</summary>

Is there an easy way to improve recall rate beacuse I find the that of baseline is not very good. What I can think of is that backbone can not use COCO's pre-training experience, but pre-training backbone for this problem which can effectively identify the foreground and background.

It is not difficult to implement pre-trained backbone under `mmdetection/tools` :

- `center_loss.py`, read the source code of `YOLOX` in view of the small increase in recall rate, it is believed that the distinction of front background features extracted by backbone is not obvious and then leading to neck and head behind may consider background features as foreground, the foreground feature is the background. Thus, Center Loss is used to increase the distinction expression. The accuracy of pre-background discrimination was 96.67%, +5.3% mAP, +3.2% mAR. Ablation experiments show that Center Loss is better than Cross entropy loss alone.
- `pretrain.py`, load pretrained backbone during detection

</details>

## Strategy two, CIoU Loss to correct defects of SimOTA

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.379
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.835
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.291
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.248
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.515
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.617
```

<details><summary>Details</summary>

However, strategy 1 also brings a problem. The detection effect of Yolox-Tiny is not significantly improved by using this strategy, and the detection effect of Yolox-Tiny is better than that of Yolox-S 3.2%mAP. 

After reading the source code, it is found that YOLOX's SimOTA mechanism has some bugs when allocating positive samples to small objects. Please refer to the link on the right side of the repo for detailed analysis. In short, the positive sample selected does not intersect with the real object because the object is small, so the Loss of CLS and OBJ is no problem, but the Loss of REG always is 1 which is unreasonable. CIoU Loss is used for correction and the effect is obviously improved.

</details>

## Strategy 3: Use self-supervised pre-training with unlabeled data

In `mmdetection/tools/ssl.py`:

```log
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.383
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.819
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.303
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.525
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.657
```

<details><summary>Details</summary>

In the real world, not all data is labeled. So how to make good use of unlabeled data? Based on the paper I read before, LET's talk about:

- Semi-supervised object detection, Microsoft published a SOTA related article in ICCV 2021, but the parameters are complicated and the model capacity needs to be doubled, which is not friendly to non-RMB players
- Object detection in self-supervised area, DetCo improved based on Moco which paper and code I read and found are not friendly for non-RMB players, and Moco and Simsiam ideas from Facebook AI research are strange and simple, but not easy to accept
- The baseline of self-EMD is BYOL, and the formula derivation in it is also nice, but the self-monitoring network structure in the early years is not concise

In conclusion, is there a simple self-supervised training method for object detection in specific scenarios? Inspired by self-EMD, I did the following simple works:

<p align="center">
    <img src="./sample/ssl.jpg" width="600">
</p>

- Cut out several patches in the picture, the blue in the middle is regarded as anchor, pink is the positive sample, and purple is the negative sample
- Using cosine distance as the loss function, the representation of Anchor and positive sample should be close, while the representation of Anchor and negative sample should be far away
- Considering that object detection is greatly affected by spatial information, patch of positive sample must be adjacent to anchor

The experimental results show that the pre-training method is superior to the labeled training method. Here I only give my thinking: for the training mode with labels, the network only recognizes the background and object. The network is only interested in the object region when throws a complete picture. If it is self-supervised, the network knows the distribution of data, or what the picture should look like, and is not particularly interested in any particular region. However, when the detection program starts to train and needs to be interested in certain region, the network will know which region it needs to be interested in, which region are similar to the region of interest, and which region are not similar to the region of interest, so that it can better locate the object.

</details>

# Thanks

Supported by High-performance Computing Platform of XiDian University.
