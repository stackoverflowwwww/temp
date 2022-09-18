import os
from PIL import Image
import numpy as np
import yaml
from u2pl.dataset import builder
import matplotlib.pyplot as plt
from u2pl.models.model_helper import ModelBuilder
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def load_state(path, model, key="state_dict"):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)

        for k in ignore_keys:
            checkpoint.pop(k)
        model.load_state_dict(state_dict, strict=False)
        best_metric = checkpoint["best_miou"]
        print("best_metric:",best_metric)
def tensor_to_img(tensor_img,mean,std):
    mean=torch.tensor(mean).reshape((-1,1,1))
    std=torch.tensor(std).reshape((-1,1,1))
    tensor_img=tensor_img*std+mean
    tensor_img=tensor_img.permute((1,2,0))
    return tensor_img.numpy().astype("int")
from u2pl.utils.utils import AverageMeter,intersectionAndUnion,init_log
import logging
def validate(
    model,
    data_loader,
    logger,
    cfg
):
    model.eval()

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )    
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    for step, batch in enumerate(data_loader):
        if step==1000:
            break
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)


    for i, iou in enumerate(iou_class):
        logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))

    return mIoU        
cfg = yaml.load(open("./temp/config_test.yaml", "r"), Loader=yaml.Loader)
loader=builder.build_vocloader("val",cfg,distributed=False)
loader_iter=iter(loader)
model = ModelBuilder(cfg["net"])
model_path=os.path.join(cfg['save_path'],'ckpt_best.pth')
load_state(model_path,model,key="teacher_state")
for p in model.parameters():
    p.requires_grad = False
model=model.cuda()
logger = init_log("global", logging.INFO)
logger.info("start evaluation")
mIou=validate(model,loader,logger,cfg)
print("mIou:",mIou)


# model.eval()
# images,labels=loader_iter.next()
# images=images.cuda()
# with torch.no_grad():
#     outs = model(images)

# output = outs["pred"]
# output = F.interpolate(
#         output, labels.shape[1:], mode="bilinear", align_corners=True
#     )
# pred=torch.argmax(output[0],0)
# pred=pred.cpu().numpy()
# cmap=color_map()
# pred_vis=cmap[pred]
# f,axs=plt.subplots(1,3)
# image=tensor_to_img(images[0].cpu(),cfg["dataset"]["mean"],cfg["dataset"]["std"])
# axs[0].imshow(image)
# labels_vis=cmap[labels[0].cpu().numpy()]
# axs[1].imshow(labels_vis)
# axs[2].imshow(pred_vis)
# plt.show()