import torch, torchvision, random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import math

'''
2018/07/04

1. RandomRotate should not be front of RandomCrop

2018/07/05

1. add pyramid, should be behind all transform

2018/08/01

1. add random grayscale, only can be used in 3 channel input network
   default probability is 0.5

2. add RandomResizedCrop, copied from pytorch, only added apply identical action for mask

'''

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img, mask):
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, image, mask):
        image = TF.resize(image, size = self.size)
        mask = TF.resize(mask, size = self.size)
        return image, mask

class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, image, mask):
        angle = transforms.RandomRotation.get_params((-self.angle, self.angle))
        mask = mask.rotate(angle)
        image = image.rotate(angle)
        return image, mask

class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
    def __call__(self, image, mask):
        w, h = image.size
        tw, th = self.crop_size

        if w == tw and h == th:
            return image, mask            
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        image = TF.crop(image, i, j, th, tw)
        mask = TF.crop(mask, i, j, th, tw)

        return image, mask

class RandomGrayscale(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return TF.to_grayscale(image, num_output_channels=3), mask
        return image, mask

class RandomResizedCrop(object):

    def __init__(self, size=320, scale=(0.08, 1.0), ratio=(1. / 2., 2. / 1.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, scale, ratio):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, mask):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        out_img =  TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        out_mask = TF.resized_crop(mask, i, j, h, w, self.size, self.interpolation)
        return out_img, out_mask

class ToGray(object):
    def __call__(self, image,mask):
        image = image.convert("L")
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask):
        return TF.to_tensor(image), TF.to_tensor(mask)

class pyramid(object):
    """
        if image size is 256*256, then pyramid(3, 2, image = True)
        will return list image = [image1, image2, image3]
	                 size is [256*256, 128*128, 64*64]
    """
    def __init__(self, num_layers, ratio, image = False, mask = False):
        self.num_layers = num_layers
        self.ratio = ratio
        self.image_T = image
        self.mask_T = mask
    def __call__(self, image, mask):
        w, h = mask.size
        if self.image_T:
            orig_image = image
            image = []
            image.append(TF.to_tensor(orig_image))
        else:
            image = TF.to_tensor(image)
        if self.mask_T:
            orig_mask = mask
            mask = []
            mask.append(TF.to_tensor(orig_mask))
        else:
            mask = TF.to_tensor(mask)
        for layer_idx in range(self.num_layers - 1):
            w //= self.ratio
            h //= self.ratio
            if self.image_T:
                image.append(TF.to_tensor(orig_image.resize((w, h), Image.BILINEAR)))
            if self.mask_T:
                mask.append(TF.to_tensor(orig_mask.resize((w, h), Image.BILINEAR)))
        return image, mask
