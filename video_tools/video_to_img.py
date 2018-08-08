import cv2, argparse, os
import numpy as np

       
parser = argparse.ArgumentParser()
parser.add_argument("--videos", help="name list of video, should be frame and mask", type = str, nargs='*' )
parser.add_argument("--outpath", help="output path", type = str, default = './video_to_img' )

args = parser.parse_args()
print (args)

num_of_video = len(args.videos)
if num_of_video == 0:
    raise ValueError("no video inputed")
elif num_of_video > 2:
    raise ValueError("number of input video should be 2")

found = 0
while True:
    name = "{}{:03d}".format(args.outpath, found)
    if os.path.exists(name):
        found += 1
    else:
        os.mkdir(name)
        break


caps = []
for video in args.videos:
    caps.append( cv2.VideoCapture(video) )

frame_path = os.path.join(name, "image")
mask_path = os.path.join(name, "mask")
os.mkdir(frame_path)
os.mkdir(mask_path)


cnt = 0
while True:
    ret, frame = caps[0].read()
    ret, mask = caps[1].read()
    if ret:
        cv2.imwrite(os.path.join(frame_path, "{}.jpg".format(cnt) ), frame)
        cv2.imwrite(os.path.join(mask_path, "{}.jpg".format(cnt) ), mask)
        cnt += 1
    else:
        break


for cap in caps:
    cap.release()
