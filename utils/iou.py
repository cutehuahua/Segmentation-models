import numpy as np
import torch
from PIL import Image


def overlap(pred, gt, gpu=False):
    
    one = torch.ones(pred.shape, requires_grad=False).cuda() if gpu else torch.ones(pred.shape, requires_grad=False)
    zero = 1 - one
    o1 = torch.where(pred != 0, one, zero)
    o2 = torch.where(gt != 0, one, zero)
    o = torch.where( (o1 + o2) == 2, one, zero)
    return o

def union(pred, gt, gpu=False):
 
    one = torch.ones(pred.shape, requires_grad=False).cuda() if gpu else torch.ones(pred.shape, requires_grad=False)
    zero = 1 - one
    u1 = torch.where(pred != 0, one, zero)
    u2 = torch.where(gt != 0, one, zero)
    u = torch.where( (u1 + u2) >= 1, one, zero)
    return u

def IOU(pred, gt, gpu=False):
    i = overlap(pred, gt, gpu)
    u = union(pred, gt, gpu)
    return i.sum() / u.sum()

'''
a = torch.Tensor(3,3).data.fill_(1)
b = torch.Tensor(3,3).data.fill_(0)
b[1, 1] = 1

print (a, b)
print (IOU(a, b))
'''

