import os
import torch.utils.data as Data
from PIL import Image

class SegmentationData(Data.Dataset):
    def __init__(self, root_path, transforms_compose = None):

        self.file_names = {}

        for each_root in root_path:
            image_root = os.path.join(each_root, "image")
            for root, dirs, files in os.walk( image_root ):
                for file in files:
                    if file.endswith("jpg") or file.endswith("png"):
                        
                        name = file.split('.')[0]
                        name = os.path.join(each_root, name)

                        image = os.path.join(each_root, "image/{}".format(file))
                        self.file_names[name] = [image]

            mask_root = os.path.join(each_root, "mask")
            for root, dirs, files in os.walk( mask_root ):
                for file in files:
                    if file.endswith("jpg") or file.endswith("png"):

                        name = file.split('.')[0]
                        name = os.path.join(each_root, name)

                        mask = os.path.join(each_root, "mask/{}".format(file))
                        self.file_names[name].append(mask)
            self.keys = []
            for key in self.file_names:
                self.keys.append(key)

        self.transforms_compose = transforms_compose

    def __getitem__(self, index):
        img = Image.open( self.file_names[ self.keys[index] ][0] )
        mask = Image.open( self.file_names[ self.keys[index] ][1] ).convert("L")

        if self.transforms_compose != None:
            img, mask = self.transforms_compose(img, mask)  
        return img, mask

    def __len__(self):
        return len(self.keys)


def get_dataset( root_path, transforms = None, train = True ):


    if type(root_path) is str:
        choose = "train" if train else "test"      
        root_path = [ os.path.join(root_path, choose ) ]

    elif type(root_path) is list:
        roots = []
        choose = "train" if train else "test"
        for root in root_path:
            roots.append( os.path.join(root, choose) )
        root_path = roots

    dataset = SegmentationData(
        root_path,
        transforms
    )

    return dataset

