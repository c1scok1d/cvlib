import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
 
#url='http://10.0.0.236/cam-hi.jpg' # front room cam
url='http://10.0.0.96/cam-hi.jpg' # lab window cam

im=None

        
def run1():
    #cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgnp,-1)
 
        # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        print(bbox, label, conf)

        # draw bounding box over detected objectsP
        out = draw_bbox(frame, bbox, label, conf)

        # display output
        frames = cv2.resize(out, (820,460))
        cv2.imshow("Object Detection", frames)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
 
 
 
if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
            f1= executer.submit(run1)