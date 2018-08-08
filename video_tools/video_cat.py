import cv2, argparse, os
import numpy as np



def get_video_writer(file_name, cat = -1, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    found = 0
    while True:
        if os.path.isfile("./{}{:03d}.avi".format(file_name, found)):
            found += 1
        else:
            break
    assign_num = found 
    return cv2.VideoWriter('./{}{:03d}.avi'.format(file_name, assign_num) ,fourcc, float(fps), (320*cat, 320))
       
parser = argparse.ArgumentParser()
parser.add_argument("--videos", help="name list of video", type = str, nargs='*' )
parser.add_argument("--fps", help="default is 20", type = float, default = 20 )

args = parser.parse_args()
print (args)

num_of_video = len(args.videos)
if num_of_video == 0:
    raise ValueError("no video inputed")


writer = get_video_writer("cat_output", num_of_video, fps=args.fps)

caps = []
for video in args.videos:
    caps.append( cv2.VideoCapture(video) )

while True:
    imgs = []
    for i in range(num_of_video):
        ret, frame = caps[i].read()
        if ret:
            imgs.append(frame)
            end = False
        else:
            end = True
    if end:
        break

    cat = np.concatenate( imgs, axis=1 )
    writer.write(cat)

    cv2.imshow("test", cat)
    cv2.waitKey( int(1000//args.fps) )

for cap in caps:
    cap.release()
writer.release()
