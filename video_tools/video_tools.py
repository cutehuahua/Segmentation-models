import cv2, torch, argparse, os, time
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

#change model here!
from deeplab_os_8or16 import DeepLabv3_plus

'''
2018/08/07 TODO
    change model by argparse
'''

class identitical(object):
    def __call__(self, thing):
        return thing
    def write(self, thing, mask=None):
        return
    def release(self):
        return

class concatenate_video(object):
    def __init__(self, v1):
        self.v1 = v1
    def write(self, img, mask):
        cat = np.concatenate((img, mask), axis = 1)
        self.v1.write(cat)
    def release(self):
        self.v1.release()

class seperated_video(object):
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    def write(self, img, mask):
        self.v1.write(img)
        self.v2.write(mask)
    def release(self):
        self.v1.release()
        self.v2.release()

def get_video_writer(file_name, cat = False, assign_num = None, fps=25):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if assign_num == None:
        found = 0
        while True:
            if os.path.isfile("./{}{:03d}.avi".format(file_name, found)):
                found += 1
            else:
                break
        assign_num = found 

    if cat:
        return cv2.VideoWriter('./{}{:03d}.avi'.format(file_name, assign_num) ,fourcc, float(fps), (320*2, 320)), assign_num
    else:
        return cv2.VideoWriter('./{}{:03d}.avi'.format(file_name, assign_num) ,fourcc, float(fps), (320, 320)), assign_num
       
parser = argparse.ArgumentParser()
parser.add_argument("--demo", help="no stands for no model, otherwise input a checkpoint path", type = str, default = "no")
parser.add_argument("--record", help="yes stands for record mode on", type = str, default = "yes" )
parser.add_argument("--live", help="yes stands for live demo, otherwise input video name", type = str, default = "yes" )
parser.add_argument("--cat", help="yes for frame and mask output concatenated, otherwise output 2 file", type = str, default = "yes" )
parser.add_argument("--webcam", help="input webcam number, default is 0", type = int, default = 0 )
parser.add_argument("--fps", help="default is 25", type = float, default = 25 )
parser.add_argument("--os", help="output stride, default is 16", type = int, default = 16 )
parser.add_argument("--empty_mask", help="whether you need empty mask or not, default is no", type = str, default = "no")
parser.add_argument("--heat_map", help="show heat map of mask, default is no", type = str, default = "no")

args = parser.parse_args()
print (args)

if args.empty_mask != "no" and args.demo != "no":
    raise ValueError("demo mode cannot generate empty mask")

#if live mode then read webcam with assign webcam number
if args.live.lower() == "yes":
    cap = cv2.VideoCapture(args.webcam)
else:
    cap = cv2.VideoCapture(args.live)
    if not cap.isOpened():
        raise IOError("cannot open file {}".format(args.live) )

#if there is a model check point input, then use this model
if args.demo.lower() != "no":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=1, output_stride=args.os)
    model.load_state_dict(torch.load( args.demo ))
    model.cuda()

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
else:
    model = identitical()

#if record mode is on and no model is running, then record frame only
if args.record.lower() == "yes" and args.demo.lower() == "no" and args.empty_mask == "no":
    frame_recorder, _ = get_video_writer("only_frame", fps=args.fps)

elif args.record.lower() == "yes" and args.demo.lower() == "no" and args.empty_mask != "no":
    if args.cat.lower() == "yes":
        v1, _ = get_video_writer("with_empty_mask", cat = True, fps=args.fps)
        frame_recorder = concatenate_video(v1)
    else:
        v1, num = get_video_writer("frame", fps=args.fps)
        v2, _ = get_video_writer("empty_mask", fps=args.fps, assign_num=num)
        frame_recorder = seperated_video(v1, v2)

#if record mode is on and model is running, record frame and mask depends on cat 
elif args.record.lower() == "yes" and args.demo.lower() != "no":
    if args.cat.lower() == "yes":
        v1, _ = get_video_writer("with_mask", cat = True, fps=args.fps)
        frame_recorder = concatenate_video(v1)
    else:
        v1, num = get_video_writer("frame", fps=args.fps)
        v2, _ = get_video_writer("mask", assign_num = num, fps=args.fps)
        frame_recorder = seperated_video(v1, v2)
else:
    frame_recorder = identitical()


#main part
cnt, avg_inference = 0, 0

while(True):
    try:
        ret, frame = cap.read()
        if ret == False:
            break
        cv_img = cv2.resize(frame, (320, 320))
    except:
        break

    if args.demo != "no":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame)
        pil_frame = TF.resize(pil_frame, (320, 320))
        img = TF.to_tensor(pil_frame).unsqueeze(0)

        t0 = time.time()
        output = model(img.cuda())
        avg_inference += (time.time() - t0)
        cnt += 1

        zero = torch.zeros((1,320,320), requires_grad=False)
        one = torch.ones((1,320,320), requires_grad=False)

        front_mask = torch.sigmoid(output.cpu()).squeeze(0)
        output_mask = torch.where( front_mask > 0.5, one, zero)
        mask = TF.to_pil_image(output_mask)

        cv_mask = cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2BGR)
        frame_recorder.write(cv_img, cv_mask)

        if args.heat_map != "no":
            heat_map = cv2.cvtColor(np.array(TF.to_pil_image(front_mask)), cv2.COLOR_GRAY2BGR)
            cv2.imshow("heat", heat_map)

        cat = np.concatenate((cv_img, cv_mask), axis = 1)
        cv2.imshow("test", cat)
    elif args.demo == "no" and args.empty_mask != "no":
        cv_mask = np.zeros( cv_img.shape ).astype(np.uint8)
        frame_recorder.write(cv_img, cv_mask)
        cat = np.concatenate((cv_img, cv_mask), axis = 1)
        cv2.imshow("test", cat)

    else:
        cv2.imshow("test", cv_img)
        frame_recorder.write(cv_img)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

if cnt != 0:
    print (avg_inference / cnt)

cap.release()
frame_recorder.release()
