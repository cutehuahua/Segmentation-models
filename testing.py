import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn
import math, time


from models.deeplab import deeplabv3p
from utils.iou import IOU
from utils.dataloader_utils import *
from utils import dataloader

import cv2
import numpy as np
import torchvision.transforms.functional as TF
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="input checkpoint path", type = str)
parser.add_argument("--mask", help="show mask or not, default is no", type = str, default = "no")
args = parser.parse_args()

model = deeplabv3p(input_channel=3, num_class=1, output_stride=16)
m = "deeplab_os16"
model.load_state_dict(torch.load( args.model ))
model.cuda()
model.eval()

transform_crop = Compose([
    #ToGray(),
    RandomGrayscale(p = 1),
    Resize((320, 320)),
    ToTensor()
])

dataset = dataloader.get_dataset(
    "/home/hua/Desktop/dataset/ae_hand",
    transform_crop,
    train = False
)

test_loader = Data.DataLoader(dataset, batch_size = 1, shuffle = True, num_workers = 4)
mean_iou, count, mean_spent, showing = 0, 0, 0, 0

s0, s1, s2 = 0, 0, 0
t0 = time.time()
for img, mask in test_loader:
    mask_shape = mask.shape
    break

with torch.no_grad():
    zero = torch.zeros(mask_shape)
    one = torch.ones(mask_shape)

    for img, mask in test_loader:
        s0 += (time.time() - t0)

        mask.requires_grad = False
        img.requires_grad = False

        t0 = time.time()
        output = model(img.cuda()).cpu()
        mean_spent += (time.time() - t0)
            
        front_mask = torch.sigmoid(output)
        output_mask = torch.where( front_mask > 0.5, one, zero)

        if args.mask != "no":
            show_mask = time.time()
            pil_mask = TF.to_pil_image(output_mask.squeeze(0))
            cv_mask = cv2.cvtColor(np.array(pil_mask), cv2.COLOR_GRAY2BGR)
            cv2.imshow("test", cv_mask)
            cv2.waitKey(1)
            showed = time.time()

            showing += (showed - show_mask)

        s1 += (time.time() - t0)   
        iou = float(IOU(output_mask, mask))
        if not math.isnan(iou):
            mean_iou += iou
            count += 1
        s2 += (time.time() - t0)


    mean_iou /= count

    mean_spent /= len(test_loader)
    s0 /= len(test_loader) 
    s1 /= len(test_loader) 
    s2 /= len(test_loader)
    showing/= len(test_loader)

    print ("model :{}\tmean IOU :{}".format(m, mean_iou))
    print ("inference : {}\ngenerate mask : {}\ncompute iou : {}\nload next batch : {}\ntotal : {}\n".format(
                    mean_spent,
                    s1 - mean_spent - showing, 
                    s2 - s1, 
                    s0 - s2,
                    s0
    ))

    if args.mask != "no":
        print ("showing mask : {}".format(
                        showing, 
        ))
