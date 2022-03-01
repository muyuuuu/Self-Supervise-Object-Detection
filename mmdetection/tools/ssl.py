# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
from operator import neg, pos
import os
import os.path as osp
from re import S
import time
import warnings

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import random
import mmcv
import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_backbone
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="resume from the latest checkpoint automatically",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="(Deprecated, please use --gpu-id) number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="(Deprecated, please use --gpu-id) ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both "
            "specified, --options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args


def get_model():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn(
            "`--gpus` is deprecated because we only support "
            "single GPU mode in non-distributed training. "
            "Use `gpus=1` now."
        )
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(
            "`--gpu-ids` is deprecated, please use `--gpu-id`. "
            "Because we only support single GPU mode in "
            "non-distributed training. Use the first GPU "
            "in `gpu_ids` now."
        )
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f"Set random seed to {seed}, " f"deterministic: {args.deterministic}")
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(args.config)

    model = build_backbone(cfg.model)
    model = model.backbone
    model.init_weights()
    return model


class mymodel(torch.nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.backbone = get_model()
        self.pool = torch.nn.AvgPool2d(kernel_size=2)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        _, _, y = self.backbone(x)
        y = self.conv(y)
        y = self.pool(y)
        y = y.view(y.size(0), -1)
        return y


class DATA(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = []
        self.size = 0
        self.img_size = 224

        self.create_data()
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.trainform = transforms.Compose(
            [
                lambda x: Image.open(x).convert("RGB"),  # open image
                transforms.Resize((720, 1280)),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),  # Converts a PIL Image to [0, 1]
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def create_data(self):
        for file in os.listdir(self.root):
            self.imgs.append(file)
        self.size = len(self.imgs)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        file = os.path.join(self.root, self.imgs[idx])
        img = self.trainform(file)
        _, h, w = img.size()
        pos_x = random.randint(0, 200)
        pos_y = random.randint(0, h - self.img_size * 3 - 1)

        pos = [(0, 1), (1, 0), (1, 2), (2, 1)]
        random.shuffle(pos)
        pos_img = [
            img[
                :,
                pos_y + y * self.img_size : pos_y + (y + 1) * self.img_size,
                pos_x + x * self.img_size : pos_x + (x + 1) * self.img_size,
            ]
            for x, y in pos
        ]
        neg = [
            (
                random.randint(w - self.img_size * 2, w - self.img_size - 1),
                random.randint(0, h - self.img_size - 1),
            )
            for _ in range(4)
        ]
        neg_img = [img[:, y : y + self.img_size, x : x + self.img_size] for x, y in neg]
        img = img[
            :,
            pos_y + self.img_size : pos_y + 2 * self.img_size,
            pos_x + self.img_size : pos_x + 2 * self.img_size,
        ]
        return img, pos_img, neg_img


if __name__ == "__main__":
    model = mymodel().cuda()
    data = DATA("/home/20031211375/flow/ssl/")
    length = len(data)
    train_length = int(length * 0.8)
    val_length = int(length * 0.2)
    train_set, val_set = torch.utils.data.random_split(data, [train_length, val_length])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}], lr=1e-3, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)

    criterion = nn.CosineSimilarity(dim=1).cuda()

    model_pth = "backbone.pth"
    if os.path.isfile(model_pth):
        checkpoint = torch.load(model_pth)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        print("load checkpoint")
    else:
        start_epoch = 1
        print("=> no checkpoint")

    max_loss = 1000000000
    for start_epoch in range(10):
        model.train()
        all_loss = []
        for idx, (x, pos_, neg_) in enumerate(train_loader):
            x = x.cuda()

            y = model(x)
            y = y.repeat(4, 1)
            pos_ = torch.cat(pos_, 0).cuda()
            neg_ = torch.cat(neg_, 0).cuda()
            y1 = model(pos_)
            y2 = model(neg_)
            ls1 = criterion(y, y1).mean()
            ls2 = criterion(y, y2).mean()
            print(idx, ls1, ls2)
            loss = -ls1 + ls2
            all_loss.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_str = time.asctime(time.localtime(time.time()))
        avg_loss = sum(all_loss) / len(all_loss)
        print("{}, Epoch: {}, loss: {:.4f}".format(time_str, start_epoch, avg_loss))

        checkpoints_best = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": start_epoch,
        }
        if avg_loss < max_loss:
            max_loss = avg_loss
            torch.save(checkpoints_best, "backbone.pth")
            print("Save Model")
        all_loss = []
