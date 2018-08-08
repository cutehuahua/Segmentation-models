import torch.nn as nn
import torch

import torch.utils.data as Data
from torchvision.utils import save_image

import models
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="name for this training", type = str)
parser.add_argument("--color", help="input color, default if RGB", type = str, default = "RGB")
parser.add_argument("--pretrained", help="input checkpoint path", type = str, default = "none")

parser.add_argument("--data_root", help="input path of main data root", type = str, default = "~/dataset/")


parser.add_argument("--os", help="output stride, default is 16", type = int, default = 16 )
parser.add_argument("--batch_size", help="batch size, default is 8", type = int, default = 8 )
parser.add_argument("--subbatch", help="subbatch, default is 1", type = int, default = 1 )


args = parser.parse_args()
print (args)


batch_size = 8
subbatch = 1 

save_each_epoch = 1
max_epoch = 10
start_epoch = 0

model = models.deeplabv3p(num_class=1, input_channel=3, output_stride=16).cuda()
m = args.name

pretrain = True
if pretrain:
    model.load_state_dict(torch.load("./saved_model/coco_voc_pretrained/deeplab_os16_100.tar"))

transform_crop = Compose([
    #ToGray(),
    Resize((320, 320)),
    #RandomBlur(p = 0.5),
    #RandomResizedCrop(),
    RandomGrayscale(),
    RandomRotate(30),
    ToTensor()
])


print ("loading data...")
dataset = dataloader.get_dataset(
    "/home/hua/Desktop/dataset/ae_hand",
    transform_crop,
    train = True
)

train_loader = Data.DataLoader(dataset, batch_size = (batch_size//subbatch), shuffle = True, num_workers = 4)
print ("done!")

model.cuda()

optim = torch.optim.Adam(model.parameters(), 1e-4)
crit = nn.BCEWithLogitsLoss()

for epoch in range(start_epoch, max_epoch + 1):

    model.train()
    for ix, (img, mask) in enumerate(train_loader):

        output = model(img.cuda())
        loss = crit(output, mask.cuda())
        loss.backward()

        if ix % subbatch == 0:
            optim.step()
            optim.zero_grad()

        if ix % (10 * subbatch) == 0:
            print ( "model:{}\tepoch:{:03d}\titeration:{:03d}\tloss:{}".format(m, epoch, (ix//subbatch), float(loss.data)) )

        if ix % (100 * subbatch) == 0:
            zero = torch.zeros(mask.shape)
            one = torch.ones(mask.shape)
            front_mask = torch.sigmoid(output.cpu())
            output_mask = torch.where( front_mask > 0.5, one, zero)

            save_image( torch.cat((output_mask, mask), dim=0).data, "./training_result/{}_{:03d}_{:03d}_mask.jpg".format(m, epoch, (ix//subbatch)), nrow = batch_size//subbatch )
            save_image( img.data, "./training_result/{}_{:03d}_{:03d}_img.jpg".format(m, epoch, (ix//subbatch)), nrow = 20//subbatch )

    if epoch % save_each_epoch == 0:
        torch.save(  model.state_dict() , "./saved_model/{}_{}.tar".format( m, epoch))


