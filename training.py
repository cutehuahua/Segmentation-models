import torch.nn as nn
import torch

import torch.utils.data as Data
from torchvision.utils import save_image

from models.deeplab import deeplabv3p
from utils import parser, dataloader
import argparse

p = argparse.ArgumentParser()
p.add_argument("--path", help="config file path", type = str)
args = p.parse_args()
args = parser.parse_config( args.path )
print (args)

batch_size = int(args.batch_size)
subbatch = int(args.subbatch)
num_of_stage = int(args.num_of_stage)
save_img_every_iterations = int(args.save_img_every_iterations)
save_each_iteration = int(args.save_each_iteration)
device = torch.device("cuda:0" if args.gpu=="yes" else "cpu")

transforms, stage_iterations, datapaths = [], [], []
for i in range(num_of_stage):
    stage_data = getattr(args, "stage{}".format(i) ).split(',')
    stage_argumentation = getattr(args, "data_argument{}".format(i) )

    datapaths.append( stage_data[:-1] ) 
    stage_iterations.append( int(stage_data[-1]) )

    transform = parser.get_transform(stage_argumentation)
    transforms.append(transform)

model = deeplabv3p(input_channel= 3 if args.color.lower() == "rgb" else 1, num_class = 1, output_stride= int(args.os)).to(device)
pretrain = ( args.pretrained != None)
if pretrain:
    model.load_state_dict(torch.load( args.pretrained ))

print ("loading data...")
train_loaders = []
for i in range(num_of_stage):
    tmp = dataloader.get_dataset(
        datapaths[i],
        transforms[i],
        train = ( args.train.lower() == "yes" )
    )
    train_loaders.append( Data.DataLoader( tmp, batch_size = (batch_size//subbatch), shuffle = True, num_workers = 4) )

if len(stage_iterations) != len(train_loaders):
    raise ValueError("wrong setting of stage iterations")
print ("done!")

model.to(device)
optim = torch.optim.Adam(model.parameters(), args.lr)
crit = nn.BCEWithLogitsLoss()
model.train()

iteration = 0
stage = 0
train_loader = train_loaders[stage]
while True:

    for ix, (img, mask) in enumerate(train_loader):

        output = model(img.to(device) )
        loss = crit(output, mask.to(device))
        loss.backward()

        if ix % subbatch == 0:
            optim.step()
            optim.zero_grad()
            iteration += 1

        if iteration % 10 == 0:
            print ( "model:{}\t\titeration:{:05d}\tloss:{}".format(args.name, iteration, float(loss.data)) )

        if iteration % save_img_every_iterations == 0:
            zero = torch.zeros(mask.shape)
            one = torch.ones(mask.shape)
            front_mask = torch.sigmoid(output.cpu())
            output_mask = torch.where( front_mask > 0.5, one, zero)

            save_image( torch.cat((output_mask, mask), dim=0).data, "./training_result/{}_{:05d}_mask.jpg".format(args.name, iteration, nrow = batch_size//subbatch ))
            save_image( img.data, "./training_result/{}_{:05d}_img.jpg".format(args.name, iteration, nrow = 20//subbatch ))

        if iteration % save_each_iteration == 0:
            torch.save(  model.state_dict() , "./saved_model/{}_{:05d}.tar".format( args.name, iteration))

        if iteration >= sum(stage_iterations):
            break

        if iteration >= sum(stage_iterations[:stage + 1]) :
            stage += 1
            train_loader = train_loaders[stage]
            break

    if iteration >= sum(stage_iterations):
        break


